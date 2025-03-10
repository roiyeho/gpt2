import torch
from torch.utils.data import Dataset, DataLoader
from bpe_tokenizer import BPETokenizer
import requests
import os

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Load and tokenize the dataset
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokens = tokenizer.encode(text)
        
        # Create input sequences
        self.data = [
            self.tokens[i:i + seq_length] 
            for i in range(0, len(self.tokens) - seq_length, seq_length)
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        target_seq = torch.tensor(self.data[idx][1:], dtype=torch.long)
        return input_seq, target_seq


# Function to get DataLoader
def get_dataloader(file_path, tokenizer, batch_size=8, seq_length=128):
    dataset = TextDataset(file_path, tokenizer, seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
