import os
import requests
from bpe_tokenizer import BPETokenizer

class DatasetManager:
    def __init__(self, dataset_path="wikitext-2.txt", vocab_size=32000, seq_length=128, batch_size=8):
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.tokenizer = BPETokenizer(vocab_size)
        self.corpus_path = None

    def load_dataset(self, path="wikitext-2.txt"):
        if not os.path.exists(path):
            self.download_dataset()
        return path
            
    # Function to download a dataset if not available
    def download_dataset(self, path="wikitext-2.txt"):
        url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
        response = requests.get(url)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Dataset downloaded and saved to {path}")
        
    def train_tokenizer(self, corpus_path):        
        print("Training BPE tokenizer...")
        self.tokenizer.train(corpus_path)
        self.tokenizer.save()

    def load_tokenizer(self):
        if os.path.exists("bpe_tokenizer.json"):
            print("Loading pre-trained tokenizer...")
            self.tokenizer.load()
        else:
            print("Tokenizer does not exist")
            
    def get_dataloaders(self):
        """Returns training and validation DataLoaders."""
        #dataloader = get_dataloader(self.dataset_path, self.tokenizer, self.batch_size, self.seq_length)
        #return dataloader
        pass
