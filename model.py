import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import ModernBertModel  # For the BERT-based text encoder



class LSTMVIS(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMVIS, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        # print("LSTM okkkutput shape:", lstm_out.shape)
        out = self.fc(lstm_out)
        out = self.dropout(out)
        return lstm_out, out
    
class TextEncoder(nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base", trainable=True):
        super().__init__()
        self.model = ModernBertModel.from_pretrained(model_name)
        self.freeze_layers()
        self.target_token_idx = 0  # CLS token

        # Make the last 4 layers trainable
    def freeze_layers(self):
        for p in [
            *list(self.model.parameters())
        ]:  # Freeze everything except the last two layers
            p.requires_grad = False
    

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state # return the embedding of the target token
        # return last_hidden_state[:, self.target_token_idx, :] 
     # return the embedding of the target token


class ExpertsImage(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_experts=10):
        super(ExpertsImage, self).__init__()
        self.num_experts = num_experts
        # A common linear layer to produce gating weights
        self.gate = nn.Linear(embedding_size, num_experts)
        # Define expert branches; for simplicity, use linear layers here.
        self.expert_nets = nn.ModuleList([
            nn.Linear(embedding_size, hidden_size) for _ in range(num_experts)
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, seq_len, embedding_size)
        # Compute gating weights for all experts.
        gating_logits = self.gate(x)  # (batch, seq_len, num_experts)
        gating_weights = self.softmax(gating_logits)
        
        # For this example, select top2 experts.
        top2_weights, top2_indices = torch.topk(gating_weights, 2, dim=-1)  # each: (batch, seq_len, 2)
        
        # Compute expert outputs for each expert branch.
        # We'll compute for all experts and then gather the top2.
        all_expert_outputs = []
        for expert in self.expert_nets:
            # Each expert output: (batch, seq_len, hidden_size)
            all_expert_outputs.append(expert(x))
        # Stack expert outputs: (batch, seq_len, num_experts, hidden_size)
        expert_outputs = torch.stack(all_expert_outputs, dim=2)
        
        # Gather the top2 expert outputs for each token.
        # top2_indices has shape (batch, seq_len, 2). We need to expand it to gather along the expert dimension.
        top2_expert_outputs = torch.gather(
            expert_outputs, 
            dim=2, 
            index=top2_indices.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.size(-1))
        )  # (batch, seq_len, 2, hidden_size)
        
        # Return the top2 gating weights and the corresponding expert outputs.
        return top2_weights, top2_expert_outputs
class ExpertsText(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_experts=10):
        super(ExpertsText, self).__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        # A linear layer to compute gating logits from the input embeddings.
        self.gate = nn.Linear(embedding_size, num_experts)
        # Create a set of expert branches. Each expert processes the input separately.
        self.expert_nets = nn.ModuleList([
            nn.Linear(embedding_size, hidden_size) for _ in range(num_experts)
        ])
        # Softmax is applied over the expert dimension.
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embedding_size)
        Returns:
            top2_weights: Tensor of shape (batch_size, seq_len, 2) with the gating weights for the top-2 experts.
            top2_expert_outputs: Tensor of shape (batch_size, seq_len, 2, hidden_size) with the outputs of the top-2 experts.
        """
        # x shape: (batch_size, seq_len, embedding_size)
        batch_size, seq_len, _ = x.size()
        
        # Compute gating logits and then gating weights.
        gating_logits = self.gate(x)                # (batch_size, seq_len, num_experts)
        gating_weights = self.softmax(gating_logits)  # (batch_size, seq_len, num_experts)
        
        # Select top2 gating weights and the corresponding expert indices.
        top2_weights, top2_indices = torch.topk(gating_weights, 2, dim=-1)  # each: (batch_size, seq_len, 2)
        
        # Compute the outputs from all expert branches.
        expert_outputs = []
        for expert in self.expert_nets:
            # Each expert processes the input: (batch_size, seq_len, hidden_size)
            expert_outputs.append(expert(x))
        # Stack the outputs to form a tensor of shape: (batch_size, seq_len, num_experts, hidden_size)
        expert_outputs = torch.stack(expert_outputs, dim=2)
        
        # Gather the outputs corresponding to the top2 expert indices.
        # top2_indices: (batch_size, seq_len, 2) --> we need to expand it for the hidden_size dimension.
        top2_expert_outputs = torch.gather(
            expert_outputs,
            dim=2,
            index=top2_indices.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)
        )  # (batch_size, seq_len, 2, hidden_size)
        
        return top2_weights, top2_expert_outputs
    
class cross_attentive_gating_mech(nn.Module):
    def __init__(self, embedding_size, num_experts=4):
        super(cross_attentive_gating_mech, self).__init__()
        self.embedding_size = embedding_size
        self.enhance_layer_tg = nn.LayerNorm(embedding_size)
        self.enhance_layer_ig = nn.LayerNorm(embedding_size)
        self.mlp_e1 = nn.Linear(embedding_size, num_experts)
        self.soft_e1 = nn.Softmax(dim=2)
        self.mlp_e2 = nn.Linear(embedding_size, num_experts)
        self.soft_e2 = nn.Softmax(dim=2)
        self.temperature = 0.1

    def calculate_logits(self, embedding1, embedding2):
        embedding2_transposed = embedding2.transpose(1, 2)  # Shape: (batch_size, embedding_size, seq_length2)
        logits = torch.matmul(embedding1, embedding2_transposed) / self.temperature  # Shape: (batch_size, seq_length1, seq_length2)
        return logits
    
    def enhance_embed_txt(self, embedding1, embedding2, logits):
        attn_weights = torch.softmax(logits, dim=-1)  # Softmax over seq_length2
        output = torch.matmul(attn_weights, embedding2)  # Shape: (batch_size, seq_length1, embedding_size)
        enhanced_embedding = embedding1 + output
        enhanced_embedding = self.enhance_layer_tg(enhanced_embedding)
        return enhanced_embedding

    def enhance_embed_img(self, embedding1, embedding2, logits):
        attn_weights = torch.softmax(logits, dim=-1)  # Softmax over seq_length2
        output = torch.matmul(attn_weights, embedding2)  # Shape: (batch_size, seq_length1, embedding_size)
        enhanced_embedding = embedding1 + output
        enhanced_embedding = self.enhance_layer_ig(enhanced_embedding)
        # Return the enhanced embedding to avoid passing None further
        return enhanced_embedding

    def forward(self, emb_1, emb_2):
        cos_e1 = self.calculate_logits(emb_1, emb_2)
        cos_e2 = self.calculate_logits(emb_2, emb_1)
        
        enhanced_embedding_e1 = self.enhance_embed_txt(emb_1, emb_2, cos_e1)
        enhanced_embedding_e2 = self.enhance_embed_img(emb_2, emb_1, cos_e2)
        
        mlpb_e1 = self.mlp_e1(enhanced_embedding_e1)
        mlpb_e2 = self.mlp_e2(enhanced_embedding_e2)
        
        # Apply softmax to the outputs of the linear layers
        softmax_e1 = self.soft_e1(mlpb_e1)
        softmax_e2 = self.soft_e2(mlpb_e2)
        
        # Mean pooling along the experts dimension
        softmax_e1 = torch.mean(softmax_e1, dim=2, keepdim=True)
        softmax_e2 = torch.mean(softmax_e2, dim=2, keepdim=True)

        return softmax_e1, softmax_e2
class ModifiedAttentionBlock(nn.Module):
    def __init__(self, embed_dim):
        super(ModifiedAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = math.sqrt(embed_dim)

    def forward(self, mha_output, gates):
        """
        Args:
            mha_output: Tensor of shape (batch, seq_len, embed_dim) from MHA.
            gates: Tensor of shape (batch, seq_len) or (batch, seq_len, 1) computed from cross attention.
        Returns:
            gated_attn_output: The output after modulated attention.
            attn_weights: The computed attention weights.
        """
        # Ensure gates has shape (batch, seq_len, 1)
        if gates.dim() == 2:
            gates = gates.unsqueeze(-1)
        
        # Compute queries, keys, and values from the MHA output.
        Q = self.linear_q(mha_output)    # (batch, seq_len, embed_dim)
        K = self.linear_k(mha_output)    # (batch, seq_len, embed_dim)
        V = self.linear_v(mha_output)    # (batch, seq_len, embed_dim)
        
        # Integrate the gates into the query and key representations.
        # (Element-wise multiplication modulates each token's features.)
        Q_mod = Q * gates
        K_mod = K * gates
        
        # Compute attention scores with scaled dot-product.
        scores = torch.matmul(Q_mod, K_mod.transpose(-2, -1)) / self.temperature  # (batch, seq_len, seq_len)
        
        # Softmax over the keys dimension.
        attn_weights = self.softmax(scores)
        
        # Compute the final gated attention output.
        gated_attn_output = torch.matmul(attn_weights, V)  # (batch, seq_len, embed_dim)
        
        return gated_attn_output, attn_weights
    
class MOEModel(nn.Module):
    def __init__(self, num_classes=5):
        super(MOEModel, self).__init__()
        # Assume these modules are defined elsewhere:
        self.image_encoder = LSTMVIS(input_dim=35,hidden_dim=128,output_dim=768,num_layers=2)         # e.g., returns (batch, seq_len, 768)
        self.text_encoder = TextEncoder()           # e.g., returns (batch, seq_len, 768)
        self.experts_image = ExpertsImage(768, 512, num_experts=10)
        self.experts_text = ExpertsText(768, 512, num_experts=10)
        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.cross_attention = cross_attentive_gating_mech(512, num_experts=10)
        
        # Our modified attention block that integrates gates.
        self.modified_attn_block = ModifiedAttentionBlock(embed_dim=512)
        
        # A classification head for the final fused representation.
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, batch):
        features, inputs, attention_mask= batch['features'], batch['input_ids'], batch['attention_mask']
        # print("features shape",features.shape)
        # print("inputs shape",inputs.shape)
        # print("attention_mask shape",attention_mask.shape)


        _,image_embedding= self.image_encoder(features)  # (batch, seq_len, 768)
        # print("image embedding shape",image_embedding.shape)
        text_embedding = self.text_encoder(inputs,attention_mask)       # (batch, seq_len, 768)
        # print("text embedding shape",text_embedding.shape)  
        top2_weights_image, top2_expert_outputs_image = self.experts_image(image_embedding)  # (batch, seq_len, 2)
        # print("top2 experts image shape",top2_weights_image.shape)
        # print("top2 experts outputs image shape",top2_expert_outputs_image.shape)
        top2_weights_text, top2_expert_outputs_text = self.experts_text(text_embedding)
        # print("top2 experts text shape",top2_weights_text.shape)
        # print("top2 experts outputs text shape",top2_expert_outputs_text.shape)
        image_exp_emb = torch.einsum("ijk,ijkl->ijl", top2_weights_image, top2_expert_outputs_image)
        text_exp_emb = torch.einsum("ijk,ijkl->ijl", top2_weights_text, top2_expert_outputs_text)
        # print("image_exp_emb shape",image_exp_emb.shape)
        # print("text_exp_emb shape",text_exp_emb.shape)


        image_exp_emb_T = image_exp_emb.transpose(0, 1)  # (seq_len, batch, 768)
        # print("image_exp_emb_T shape",image_exp_emb_T.shape)
        text_exp_emb_T = text_exp_emb.transpose(0, 1)    # (seq_len, batch, 768)
        # print("text_exp_emb_T shape",text_exp_emb_T.shape)
        
        image_mha_out, _ = self.mha(image_exp_emb_T, image_exp_emb_T, image_exp_emb_T)
        # print("image_mha_out shape",image_mha_out.shape)
        text_mha_out, _ = self.mha(text_exp_emb_T, text_exp_emb_T, text_exp_emb_T)
        # print("text_mha_out shape",text_mha_out.shape)
        # Transpose back to (batch, seq_len, 768)
        image_exp_emb_mha = image_mha_out.transpose(0, 1)
        # print("image_exp_emb_mha shape_after",image_exp_emb_mha.shape)
        text_exp_emb_mha = text_mha_out.transpose(0, 1)
        # print("text_exp_emb_mha shape_after",text_exp_emb_mha.shape)
        
        # Compute decoding gates via cross-attention between modalities.
        # Assume cross_attention returns two tensors (for image and text) of shape (batch, seq_len).
        gates_image, gates_text = self.cross_attention(image_exp_emb, text_exp_emb)
        # print("gates_image shape",gates_image.shape)
        # print("gates_text shape",gates_text.shape)
        
        # Now integrate the MHA output with the gates via our modified attention block.
        final_image_att, attn_img = self.modified_attn_block(image_exp_emb_mha, gates_image)
        # print("final_image_att shape",final_image_att.shape)
        final_text_att, attn_text = self.modified_attn_block(text_exp_emb_mha, gates_text)
        # print("final_text_att shape",final_text_att.shape)
        
        # Fuse the final attended representations (e.g., by concatenation).
        fused_features = torch.cat([final_image_att, final_text_att], dim=1)  # (batch, seq_len, 768*2)
        # print("fused_features shape",fused_features.shape)
        
        # Pool across the sequence dimension (e.g., mean pooling).
        fused_features = fused_features.mean(dim=1)  # (batch, 768*2)
        # print("fused_features shape_after",fused_features.shape)
        
        # Classification head.

        logits = self.classifier(fused_features)  # (batch, num_classes)

        return logits


        # print("logits shape",logits.shape)

    def evaluate(self,image, inputs, attention_mask):
        _,image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(inputs,attention_mask)
        top2_weights_image, top2_expert_outputs_image = self.experts_image(image_embedding)
        top2_weights_text, top2_expert_outputs_text = self.experts_text(text_embedding)
        image_exp_emb = torch.einsum("ijk,ijkl->ijl", top2_weights_image, top2_expert_outputs_image)
        text_exp_emb = torch.einsum("ijk,ijkl->ijl", top2_weights_text, top2_expert_outputs_text)
        image_exp_emb_T = image_exp_emb.transpose(0, 1)
        text_exp_emb_T = text_exp_emb.transpose(0, 1)
        image_mha_out, _ = self.mha(image_exp_emb_T, image_exp_emb_T, image_exp_emb_T)
        text_mha_out, _ = self.mha(text_exp_emb_T, text_exp_emb_T, text_exp_emb_T)
        image_exp_emb_mha = image_mha_out.transpose(0, 1)
        text_exp_emb_mha = text_mha_out.transpose(0, 1)
        gates_image, gates_text = self.cross_attention(image_exp_emb, text_exp_emb)
        final_image_att, attn_img = self.modified_attn_block(image_exp_emb_mha, gates_image)
        final_text_att, attn_text = self.modified_attn_block(text_exp_emb_mha, gates_text)
        fused_features = torch.cat([final_image_att, final_text_att], dim=1)
        fused_features = fused_features.mean(dim=1)
        logits = self.classifier(fused_features)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        predicted_class = predicted_class.cpu().tolist()
        probabilities = probabilities.cpu().tolist()  
        return predicted_class, probabilities