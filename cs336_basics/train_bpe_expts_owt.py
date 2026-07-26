"""Module providing functions to import the Path."""

import argparse
from pathlib import Path
from cs336_basics.bpe_tokenizer import BPETokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

if __name__ == "__main__":
    # Configure the command line argiments for training the BPE algorithm on
    # OpenWebText.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data_file",
        type=str,
        required=True,
        help="The name of the training file. It should be present in 'data' folder under the assignment1-basics",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="The maximum vocabulary size to learn over the training text."
    )
    ARGS = parser.parse_args()

    tokenizer = BPETokenizer()
    input_path = DATA_DIR / ARGS.training_data_file
    vocab, merges = tokenizer.parallel_tokenize(str(input_path), vocab_size=ARGS.vocab_size,
                                                special_tokens=["<|endoftext|>"])
    vocab_path = DATA_DIR / (ARGS.training_data_file + "vocab.txt")
    merges_path = DATA_DIR / (ARGS.training_data_file + "merges.txt")
    tokenizer.materialize_vocab(vocab_path)
    tokenizer.materialize_merges(merges_path)