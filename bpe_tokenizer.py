from tokenizers import ByteLevelBPETokenizer

class BPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.tokenizer = ByteLevelBPETokenizer()        
    
    def train(self, corpus_path):
        self.tokenizer.train(
            files=[corpus_path],
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        )
        self.save()
    
    def save(self, path="."):
        self.tokenizer.save_model(path, "bpe_tokenizer")
    
    def load(self, vocab_file="bpe_tokenizer-vocab.json", merges_file="bpe_tokenizer-merges.txt"):
        """
        Loads a ByteLevel BPE tokenizer from the given vocab + merges files.
        """
        self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids):       
        return self.tokenizer.decode(token_ids)

    

