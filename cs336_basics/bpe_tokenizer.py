# import re
import pickle
import time
import regex as re

GPT_PRE_TOKENIZER_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    """
    BPE tokenizer class trains the byte pair encoding algorithm on a given dataset. It
    generates the vocabulary, and the merges done during the training to generated the
    vocabulary.
    """

    def __init__(self) -> None:
        self._vocabulary: dict[int, bytes] = {}
        self._frequency_count: dict[tuple[bytes, ...], int] = {}
        self._token_id = 0
        self._bpe_merges: list[tuple[bytes, bytes]] = list()
        self._start_time = None
        self._end_time = None

    def create_vocabulary(self, special_tokens: list[str]) -> None:
        """
        Initialized the vocabulary dict with the initial 256 bytes and the special tokens.
        Each of these are assigned a token_id starting from 0.
        """
        for i in range(256):
            self._vocabulary[self._token_id] = bytes([i])
            self._token_id += 1
        for token in special_tokens:
            self._vocabulary[self._token_id] = token.encode("utf-8")
            self._token_id += 1

    def add_to_vocabulary(self, byte_pair: tuple[bytes, bytes]):
        """Adds a new byte pair (as a new concatenated byte) to the vocabulary and
        assigns it a new token id."""
        self._vocabulary[self._token_id] = byte_pair[0] + byte_pair[1]
        self._token_id += 1

    def str_to_bytes(self, word: str) -> tuple[bytes, ...]:
        """Convert the string to a tuple of bytes in utf-8 encoding."""
        return tuple(bytes([b]) for b in word.encode("utf-8"))

    def pre_tokenization(self, document_list: list[str]):
        """Does pre tokenization and computes the frequency counts of all the words
        and returns the frequency counts. We use the GPT PRE tokenization regex to
        split into words."""
        for document in document_list:
            # print ("Pre tokenizer")
            # print (document)
            for word in re.finditer(GPT_PRE_TOKENIZER_REGEX, document):
                if self.str_to_bytes(word.group()) in self._frequency_count:
                    self._frequency_count[self.str_to_bytes(word.group())] += 1
                else:
                    self._frequency_count[self.str_to_bytes(word.group())] = 1

    def update_top_pair_in_frequency_count(self, byte_pair: tuple[bytes, bytes]):
        """Updates the frequency count and replaces individual bytes with the top pair of bytes
        For e.g. it will replace (a, n, d): 32 with (b'an', d): 32 if (b'a', b'n') are the byte pair
        in the last merge."""
        new_byte = byte_pair[0] + byte_pair[1]
        for word_byte_tuple, frequency in list(self._frequency_count.items()):
            new_word_byte_tuple: tuple[bytes, ...] = tuple()
            i = 0
            updated = False
            while i < len(word_byte_tuple):
                if i == len(word_byte_tuple) - 1:
                    # This is the last byte. Just add to the new word byte tuple
                    new_word_byte_tuple = new_word_byte_tuple + (word_byte_tuple[i],)
                    i += 1
                elif word_byte_tuple[i] + word_byte_tuple[i + 1] == new_byte:
                    new_word_byte_tuple = new_word_byte_tuple + (word_byte_tuple[i] + word_byte_tuple[i + 1],)
                    updated = True
                    i += 2
                else:
                    new_word_byte_tuple = new_word_byte_tuple + (word_byte_tuple[i],)
                    i += 1
            if updated:
                self._frequency_count[new_word_byte_tuple] = frequency
                del self._frequency_count[word_byte_tuple]
        return

    def merge(self):
        """Runs the merge algorithm:
            1) Finds the most frequent byte pair in this iteration.
            2) Adds that byte pair to the vocabulary dict
            3) Updates the frequency dictionary with that byte pair.
            4) Adds the byte pair to the list of merges.
        This function needs to be called till we reach the vocabulary limit. Each time
        this function is called, it increases the vocabulary size by 1 and adds a new byte
        pair to the bpe merge list.
        """
        pair_count: dict[tuple[bytes, bytes], int] = {}
        for words, frequency in self._frequency_count.items():
            # Iterate over pair of bytes
            for i in range(len(words) - 1):
                pair = (words[i], words[i + 1])
                if pair not in pair_count:
                    pair_count[pair] = frequency
                else:
                    pair_count[pair] += frequency
        # Find the top pair by sorting by the frequency and if the frequency is the same, then
        # sort by the key
        sorted_pair_count = dict(sorted(pair_count.items(), key=lambda item: (item[1], item[0]), reverse=True))
        # print ("Sorted pair count")
        # print (sorted_pair_count)
        first_key, _ = next(iter(sorted_pair_count.items()))
        # Add the top pair to the vocabulary
        self.add_to_vocabulary(first_key)
        # Updates the frequency count and replaces individual bytes with the top pair of bytes
        # For e.g. it will replace (a, n, d): 32 with (b'an', d): 32
        self.update_top_pair_in_frequency_count(first_key)
        self._bpe_merges.append(first_key)
        return

    def tokenize(
        self, tokenizer_training_data_path: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Runs the bpe encoding on the given training dataset, limited by the vocab size and split
        by special_tokens. It does pre tokenization, initializes the vocabulary and calls the
        merge algorithm.
        """

        self._start_time = time.perf_counter()
        with open(tokenizer_training_data_path, "r", encoding="utf-8") as file:
            content = file.read()
            escaped_special_tokens = []
            # Escapes the special token so that regex handles the special characters in the
            # separator correctly.
            for token in special_tokens:
                escaped_special_tokens.append(re.escape(token))
            document_list = re.split("|".join(escaped_special_tokens), content)
            # Compute the frequency counts for all the words in the list of documents
            self.pre_tokenization(document_list)
            # Initialize the vocabulary of the byte pair encoding algorithm
            self.create_vocabulary(special_tokens)
            # Run the merging algorithm.
            while len(self._vocabulary) < vocab_size:
                self.merge()
            self.print_debug_string()
        self._end_time = time.perf_counter()
        execution_time = self._end_time - self._start_time
        print(f"\nExecution time: {execution_time:.6f} seconds\n", flush=True)
        return (self._vocabulary, self._bpe_merges)

    def materialize_vocab(self, vocab_path):
        """Materializes the vocabulary in binary format to the given path."""
        with open(vocab_path, "wb") as file:
            pickle.dump(self._vocabulary, file)
        print("Vocab materialized data: " + str(vocab_path))

    def materialize_merges(self, merges_path):
        """Materializes the merges in binary format to the given path."""
        with open(merges_path, "wb") as file:
            pickle.dump(self._bpe_merges, file)
        print("Merges materialized data: " + str(merges_path))

    def print_debug_string(self):
        """Helper function to print various variables to help with debugging."""
        print("Vocabulary size: " + str(len(self._vocabulary)))
        print("Frequency count table: " + str(len(self._frequency_count)))
        return
        # print ("Vocabulary: \n")
        # print (self._vocabulary)
        # print (self._bpe_merges)
