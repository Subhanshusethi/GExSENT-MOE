import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

class MOELSDataset(Dataset):
    def __init__(self, pkl_file_path, tokenizer, max_length, max_feature_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_feature_length = max_feature_length  # Max length to pad/truncate features

        # Load the pickle file once
        with open(pkl_file_path, 'rb') as f:
            self.data = pickle.load(f)

        # Define sentiment mapping
        self.sentiment_mapping = {
            -1: 0,  # very negative
             0: 1,  # neutral
             1: 2   # very positive
        }

    def __getitem__(self, index):
        item = self.data[index]  

        features = torch.tensor(item['features'], dtype=torch.float32)  # Shape: (N, 35)
        text = " ".join(item['sentences'])  # Ensure text is a string

        # Ensure polarity is a scalar value before mapping
        polarity = item['polarity']
        if isinstance(polarity, np.ndarray):  
            polarity = polarity.item()

        sentiment = torch.tensor(self.sentiment_mapping.get(polarity, 1), dtype=torch.long)  # Default to neutral

        # Pad or truncate features to max_feature_length
        num_features, feature_dim = features.shape
        padded_features = torch.zeros((self.max_feature_length, feature_dim), dtype=torch.float32)
        padded_features[:min(num_features, self.max_feature_length)] = features[:self.max_feature_length]

        # Tokenize text
        tokenized = self.tokenizer(
            text, padding="max_length", truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        )

        return {
            "features": padded_features,  # Now has a fixed shape (max_feature_length, 35)
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "sentiment": sentiment
        }

    def __len__(self):
        return len(self.data)
