from model import LSTMVIS, ExpertsImage, ExpertsText, ModifiedAttentionBlock, cross_attentive_gating_mech, MOEModel
from Dataset import MOELSDataset
from visualiizer import custom_collate_fn, PeriodicPlotter, AvgMeter, get_lr
from config import CFG
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import logging
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import gc

# Load the dataset
dataset = MOELSDataset(
    pkl_file_path='processed_train_data.pkl',
    tokenizer=AutoTokenizer.from_pretrained(CFG.text_encoder_model),
    max_length=CFG.max_length,
    max_feature_length=370
)

# Define loss function
criterion = nn.CrossEntropyLoss()

def build_loaders(pkl_path, tokenizer, mode):
    dataset = MOELSDataset(
        pkl_file_path=pkl_path,
        tokenizer=tokenizer,
        max_length=CFG.max_length,
        max_feature_length=370
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

# Training loop
def train_epoch(model, train_loader, optimizer, step, lr_scheduler=None):
    model.train()

    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader), mininterval=1, desc="Training", dynamic_ncols=False)
    history = []

    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        class_logits = model(batch).to(CFG.device)  # Model outputs logits

        # Compute the loss
        loss = criterion(class_logits, batch["sentiment"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == "batch":
            lr_scheduler.step()

        count = batch["features"].size(0)
        loss_meter.update(loss.item(), count)
        history.append(loss.item())

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter

# Validation loop
def valid_epoch(model, valid_loader):
    model.eval()

    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader), mininterval=1, desc="Validation", dynamic_ncols=False)
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        class_logits = model(batch)
        loss = criterion(class_logits, batch["sentiment"])

        count = batch["features"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter

# Test model function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img_emb = data['features'].to(CFG.device)
            inp_cov = data['input_ids'].to(CFG.device)
            att_mask = data['attention_mask'].to(CFG.device)
            trg_sen = data['sentiment'].to(CFG.device)

            predicted_classes, batch_probabilities = model.evaluate(img_emb, inp_cov, att_mask)

            target_list = trg_sen.cpu().tolist()

            for pred, true_label, probs in zip(predicted_classes, target_list, batch_probabilities):
                if pred == true_label:
                    correct += 1
                total += 1
                all_predictions.append(pred)
                all_targets.append(true_label)
                all_probabilities.append(probs)

    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Compute F1 score (micro-average is commonly used for multi-class classification)
    f1 = f1_score(all_targets, all_predictions, average='micro')
    print(f"Test F1 Score: {f1:.2f}")
    
    return all_predictions, all_targets, all_probabilities, accuracy, f1


def main():
    pkl_path_train = "processed_train_data.pkl"
    pkl_path_valid = "processed_eval_data.pkl"

    tokenizer = AutoTokenizer.from_pretrained(CFG.text_encoder_model)
    train_loader = build_loaders(pkl_path_train, tokenizer, mode="train")
    valid_loader = build_loaders(pkl_path_valid, tokenizer, mode="valid")

    model = MOEModel(num_classes=3).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    step = "epoch"
    best_loss = float('inf')

    train_history = []
    valid_history = []

    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        logging.info(f"Epoch: {epoch + 1}")

        train_loss = train_epoch(model, train_loader, optimizer, step, lr_scheduler=lr_scheduler)
        train_history.append(train_loss.avg)
        torch.save(model.state_dict(), "temp.pt")
        
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            valid_history.append(valid_loss.avg)
    
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
            logging.info("Saved Best Model!")
    
        lr_scheduler.step(valid_loss.avg)

        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}/{CFG.epochs}], Training Loss: {train_loss.avg:.4f}, Validation Loss: {valid_loss.avg:.4f}")
            logging.info(f"Epoch [{epoch+1}/{CFG.epochs}], Training Loss: {train_loss.avg:.4f}, Validation Loss: {valid_loss.avg:.4f}")

    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Training loss')
    plt.plot(valid_history, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Testing model
    model.load_state_dict(torch.load("best.pt"))
    model.to(CFG.device)
    test_loader = build_loaders("processed_eval_data.pkl", tokenizer, mode="valid")
    all_predictions, all_targets, all_probabilities,acc,f1 = test_model(model, test_loader)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test F1 Score: {f1:.2f}")

if __name__ == "__main__":
    logging.basicConfig(
        filename='training.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    logging.info("Starting training process")
    main()
    torch.cuda.empty_cache()
    gc.collect()
