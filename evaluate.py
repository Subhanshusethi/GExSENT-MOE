import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from Dataset import MOELSDataset
from config import CFG
from model import MOEModel
from visualiizer import AvgMeter
from torch.nn import CrossEntropyLoss
import numpy as np
import pickle
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        # for inputs, targets in tqdm(test_loader):
        for data in tqdm(test_loader):
            # print(data)
            img_emb = data['features'].to(CFG.device)
            inp_cov = data['input_ids'].to(CFG.device)
            att_mask = data['attention_mask'].to(CFG.device)
            trg_sen = data['sentiment'].to(CFG.device)

            predicted_classes, batch_probabilities = model.evaluate(img_emb, inp_cov, att_mask)

            # Now predicted_classes is a list of predicted indices for this batch
            # and trg_sen is a tensor of shape [batch_size].
            # Convert trg_sen to list to compare:
            target_list = trg_sen.cpu().tolist()

            # Loop over batch predictions and targets
            for pred, true_label, probs in zip(predicted_classes, target_list, batch_probabilities):
                # print(f"Predicted: {pred}, True: {true_label}, Probabilities: {probs}")
                if pred == true_label:
                    correct += 1
                total += 1
                all_predictions.append(pred)
                all_targets.append(true_label)
                all_probabilities.append(probs)

    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return all_predictions, all_targets, all_probabilities

tokenizer = AutoTokenizer.from_pretrained(CFG.text_encoder_model)
model = MOEModel(num_classes=3)
model.load_state_dict(torch.load("best.pt"))
model.to('cuda')
# test_df = testing_df
test_loader = build_loaders("processed_eval_data.pkl", tokenizer, mode="valid")
all_predictions, all_targets, all_probabilities = test_model(model, test_loader)
