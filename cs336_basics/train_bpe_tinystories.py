'''Module providing functions to import the Path.'''
from pathlib import Path
from cs336_basics.bpe_tokenizer import BPETokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

if __name__ == '__main__':
    tokenizer = BPETokenizer()
    input_path = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
    vocab, merges = tokenizer.parallel_tokenize(str(input_path), vocab_size=1000, 
                                    special_tokens=["<|endoftext|>"])
    vocab_path = DATA_DIR / "TinyStoriesV2-GPT4-train-vocab.bin"
    merges_path = DATA_DIR / "TinyStoriesV2-GPT4-train-merges.bin"
    vocab_path_text = DATA_DIR / "TinyStoriesV2-GPT4-train-vocab.txt"
    merges_path_text = DATA_DIR / "TinyStoriesV2-GPT4-train-merges.txt"
    tokenizer.materialize_vocab(vocab_path)
    tokenizer.materialize_merges(merges_path)
    tokenizer.materialize_vocab_as_text(vocab_path_text)
    tokenizer.materialize_merges_as_text(merges_path_text)