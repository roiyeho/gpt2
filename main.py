from dataset_manager import DatasetManager
from bpe_tokenizer import BPETokenizer

def main():
    dataset_manager = DatasetManager()
    corpus_path = dataset_manager.load_dataset()
    print(corpus_path)
   
    tokenizer = BPETokenizer()
    #tokenizer.train(corpus_path)
    tokenizer.load()
    print('Vocab size:', tokenizer.vocab_size)

    test = tokenizer.encode('This is a GPT-2 model')
    print(test)
    print(tokenizer.decode(test))

    dataloader = dataset_manager.get_dataloader(corpus_path, tokenizer)
    print(len(dataloader))

    # Get an iterator
    dataloader_iter = iter(dataloader)

    # Print first 3 batches
    for _ in range(3):
        input_seq, target_seq = next(dataloader_iter)
        print("Input:", input_seq)
        print("Target:", target_seq)
        print("-" * 80)
       
if __name__ == "__main__":
    main()