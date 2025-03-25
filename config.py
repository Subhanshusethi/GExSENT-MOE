import torch

class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = False
    batch_size = 4
    num_workers = 0
    learning_rate = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.4
    epochs = 10
    model_name = "openai/clip-vit-base-patch32"
    image_embedding = 768
    text_encoder_model = 'answerdotai/ModernBERT-base'
    text_embedding = 768
    max_length = 24
    pretrained = True
    trainable = False
    temperature = 1
    size =244
    num_projection_layers = 2
    projection_dim = 160
    dropout = 0.50108
    image_path = "memotion_dataset_7k/images"
    conversation_path = "memotion_dataset_7k"


