from dataset_manager import DatasetManager
from bpe_tokenizer import BPETokenizer

def main():
    dataset_manager = DatasetManager()
    corpus_path = dataset_manager.load_dataset()
    #dataset_manager.train_tokenizer(corpus_path)

    tokenizer = BPETokenizer()
    tokenizer.load()
    print('Vocab size:', tokenizer.vocab_size)

    test = tokenizer.encode('This is a GPT-2 model')
    print(test)
    print(tokenizer.decode(test))
    
if __name__ == "__main__":
    main()