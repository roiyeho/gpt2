import os
import requests
from bpe_tokenizer import BPETokenizer
from text_dataset import TextDataset
from torch.utils.data import DataLoader, random_split

class DatasetManager:
    def __init__(self, dataset_path="wikitext-2.txt", vocab_size=32000, seq_length=128, 
                 batch_size=8, val_split=0.1):
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.val_split = val_split        
        self.corpus_path = dataset_path

    def load_dataset(self):
        if not os.path.exists(self.dataset_path):
            self.download_dataset()
        return self.dataset_path
            
    # Function to download a dataset if not available
    def download_dataset(self):
        url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
        response = requests.get(url)
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Dataset downloaded and saved to {self.dataset_path}")
    
    def get_dataloaders(self):
        # Load dataset and tokenizer
        self.load_dataset()
        self.tokenizer = BPETokenizer(self.vocab_size)
        self.tokenizer.load()

        dataset = TextDataset(self.corpus_path, self.tokenizer, self.seq_length)

        """Creates and returns DataLoaders for training and validation sets."""
        dataset = TextDataset(self.dataset_path, self.tokenizer, self.seq_length)
        val_size = int(self.val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader


    
