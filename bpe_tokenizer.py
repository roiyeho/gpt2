from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
    
    def train(self, corpus_path):
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
        self.tokenizer.train([corpus_path], trainer)
    
    def save(self, path="bpe_tokenizer.json"):
        self.tokenizer.save(path)
    
    def load(self, path="bpe_tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(path)
    
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
