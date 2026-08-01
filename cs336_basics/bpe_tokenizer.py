# import re
from collections import Counter
import multiprocessing
import pickle
import time
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries

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
        # All pair of bytes with their frequencies. These pairs will be updated
        # with each merge step.
        self._pair_count: dict[tuple[bytes, bytes], int] = {}
        self._token_id = 0
        self._bpe_merges: list[tuple[bytes, bytes]] = []
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
        print ("Finished initializing vocabulary")

    def add_to_vocabulary(self, byte_pair: tuple[bytes, bytes]):
        """Adds a new byte pair (as a new concatenated byte) to the vocabulary and
        assigns it a new token id."""
        self._vocabulary[self._token_id] = byte_pair[0] + byte_pair[1]
        self._token_id += 1

    def str_to_bytes(self, word: str) -> tuple[bytes, ...]:
        """Convert the string to a tuple of bytes in utf-8 encoding."""
        return tuple(bytes([b]) for b in word.encode("utf-8"))

    def pre_tokenization(self, document_list: list[str]) -> dict[tuple[bytes, ...], int]:
        """Does pre tokenization and computes the frequency counts of all the words
        and returns the frequency counts. We use the GPT PRE tokenization regex to
        split into words."""
        frequency_count: dict[tuple[bytes, ...], int] = {}
        for document in document_list:
            for word in re.finditer(GPT_PRE_TOKENIZER_REGEX, document):
                if self.str_to_bytes(word.group()) in frequency_count:
                    frequency_count[self.str_to_bytes(word.group())] += 1
                else:
                    frequency_count[self.str_to_bytes(word.group())] = 1
        return frequency_count

    def update_top_pair_in_frequency_count(self, byte_pair: tuple[bytes, bytes]):
        """
        This function does two things:
        1) Updates the frequency count and replaces individual bytes with the top pair of bytes
            For e.g. it will replace (a, n, d): 32 with (b'an', d): 32 if (b'a', b'n') are the
            byte pair in the last merge.
        2) Updates the byte pair count as a consequence of merging the bytes in the words. All
            the byte pairs that depended on this word will get updated in the @self._pair_count
            variable.
        """
        # First delete this pair from the byte pair frequency counts
        del self._pair_count[byte_pair]
        # Then merge this byte pair in the frequency count dictionary.
        new_byte = byte_pair[0] + byte_pair[1]
        for word_byte_tuple, frequency in list(self._frequency_count.items()):
            new_word_byte_tuple: tuple[bytes, ...] = tuple()
            suffix_intersection_byte_tuple: tuple[bytes, bytes]
            prefix_intersection_byte_tuple: tuple[bytes, bytes]
            i = 0
            updated = False
            while i < len(word_byte_tuple):
                if i == len(word_byte_tuple) - 1:
                    # This is the last byte. Just add to the new word byte tuple
                    new_word_byte_tuple = new_word_byte_tuple + (word_byte_tuple[i],)
                    i += 1
                elif word_byte_tuple[i] + word_byte_tuple[i + 1] == new_byte:
                    new_word_byte_tuple = new_word_byte_tuple + (word_byte_tuple[i] + word_byte_tuple[i + 1],)
                    # Update the pair count of the previous pair which had intersection with the
                    # top_pair For e.g. Let's say the top pair is. (e, st) = 'est'
                    # (a, e) will be merged with (est) and (a, e) will be reduced it's frequency
                    # resulting from this word tuple.
                    if i > 0:
                        suffix_intersection_byte_tuple = (word_byte_tuple[i - 1], new_byte)
                        if suffix_intersection_byte_tuple in self._pair_count:
                            self._pair_count[suffix_intersection_byte_tuple] += frequency
                        else:
                            self._pair_count[suffix_intersection_byte_tuple] = frequency
                        old_prefix_byte_tuple = (word_byte_tuple[i - 1], word_byte_tuple[i])
                        if old_prefix_byte_tuple in self._pair_count:
                            self._pair_count[old_prefix_byte_tuple] -= frequency
                    # Update the pair count of the next pair which had intersection with the top
                    # pair. For e.g. Let's say the top pair is, (e, st) = 'est'
                    # (st, f) will be merged with (est) to become (est, f) and (st, f) will be
                    # reduced it's frequency resulting from this word tuple.
                    if i + 2 < len(word_byte_tuple):
                        prefix_intersection_byte_tuple = (new_byte, word_byte_tuple[i + 2])
                        if prefix_intersection_byte_tuple in self._pair_count:
                            self._pair_count[prefix_intersection_byte_tuple] += frequency
                        else:
                            self._pair_count[prefix_intersection_byte_tuple] = frequency
                        old_suffix_byte_tuple = (word_byte_tuple[i + 1], word_byte_tuple[i + 2])
                        if old_suffix_byte_tuple in self._pair_count:
                            self._pair_count[old_suffix_byte_tuple] -= frequency
                    updated = True
                    i += 2
                else:
                    new_word_byte_tuple = new_word_byte_tuple + (word_byte_tuple[i],)
                    i += 1
            if updated:
                self._frequency_count[new_word_byte_tuple] = frequency
                del self._frequency_count[word_byte_tuple]
        return

    def initialize_pair_count_from_frequency(self):
        """
        Initializes the pair of bytes along with their frequency of occurence.
        This is only done at the beginning. After that we incrementally update this dictionary.
        """
        if len(self._pair_count) > 0:
            # Pair count is already initialized from frequency.
            return
        self._pair_count = {}
        for words, frequency in self._frequency_count.items():
            # Iterate over pair of bytes
            for i in range(len(words) - 1):
                pair = (words[i], words[i + 1])
                if pair not in self._pair_count:
                    self._pair_count[pair] = frequency
                else:
                    self._pair_count[pair] += frequency
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
        self.initialize_pair_count_from_frequency()
        # Find the top pair by sorting by the frequency and if the frequency is the same, then
        # sort by the key
        sorted_pair_count = dict(sorted(self._pair_count.items(), 
                                        key=lambda item: (item[1], item[0]), reverse=True))
        top_pair, _ = next(iter(sorted_pair_count.items()))
        # Add the top pair to the vocabulary
        self.add_to_vocabulary(top_pair)
        # Updates the frequency count and replaces individual bytes with the top pair of bytes
        # For e.g. it will replace (a, n, d): 32 with (b'an', d): 32
        # This also updates the byte pair frequency with the new appended bytes.
        self.update_top_pair_in_frequency_count(top_pair)
        self._bpe_merges.append(top_pair)
        return

    def escaped_special_tokens_regex(self, special_tokens: list[str]) -> str:
        """
        Escapes the special tokens and builds a regex merging all of them.
        """
        escaped_special_tokens: list[str] = []
        # Escapes the special token so that regex handles the special characters in the
        # separator correctly.
        for token in special_tokens:
            escaped_special_tokens.append(re.escape(token))
        return "|".join(escaped_special_tokens)

    def tokenize(
        self, tokenizer_training_data_path: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Runs the bpe encoding on the given training dataset, limited by the vocab size and split
        by special_tokens. It does pre tokenization, initializes the vocabulary and calls the
        merge algorithm.
        """
        escaped_special_tokens_re = self.escaped_special_tokens_regex(special_tokens)
        self._start_time = time.perf_counter()
        with open(tokenizer_training_data_path, "r", encoding="utf-8") as file:
            content = file.read()
            document_list = re.split(escaped_special_tokens_re, content)
            # Compute the frequency counts for all the words in the list of documents
            # Update the global frequency count with the frequency computed for this document.
            self._frequency_count = dict(Counter(self._frequency_count) + Counter(self.pre_tokenization(document_list)))
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

    def parallel_pre_tokenizer(
        self,
        file_start_boundary: int,
        file_end_boundary: int,
        tokenizer_training_data_path: str,
        special_tokens_regex: str,
    ):
        '''
            Parallelize the pre tokenizer by reading a chunk of the input file instead of
            the entire file.
        '''
        with open(tokenizer_training_data_path, "rb") as file:
            file.seek(file_start_boundary)
            content = file.read(file_end_boundary - file_start_boundary).decode("utf-8", errors="ignore")
            document_list = re.split(special_tokens_regex, content)
            return self.pre_tokenization(document_list)
        return

    def parallel_tokenize(
        self, tokenizer_training_data_path: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Parallelizes the bpe encoding on the given training dataset, limited by the vocab size
        and split by special_tokens. It does pre tokenization, initializes the vocabulary and calls
        the merge algorithm.

        It does the following:
        1) Splits the input file (training data file) into multiple chunks.
        2) In parallel process, calls the parallel pretokenize to compute the frequency counts.
        3) Merge the frequency counts from the parallel processes into a single frequency count 
            dictionary.
        """
        escaped_special_tokens_re = self.escaped_special_tokens_regex(special_tokens)
        self._start_time = time.perf_counter()
        with open(tokenizer_training_data_path, "rb") as file:
            num_processes = 10

            boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            with multiprocessing.Pool(processes=num_processes) as pool:
                async_result_list = []
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    async_result = pool.apply_async(
                        self.parallel_pre_tokenizer,
                        args=(start, end, tokenizer_training_data_path, escaped_special_tokens_re),
                    )
                    async_result_list.append(async_result)
                for async_result in async_result_list:
                    self._frequency_count = dict(Counter(self._frequency_count) + Counter(async_result.get()))
                    print ("Finished pre_tokenizer: " + multiprocessing.current_process().name)

            # Initialize the vocabulary of the byte pair encoding algorithm
            self.create_vocabulary(special_tokens)
            # Run the merging algorithm.
            while len(self._vocabulary) < vocab_size:
                if len(self._vocabulary) % 1000 == 0:
                    print ("Vocabolary size: " + str(len(self._vocabulary)))
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

    def materialize_vocab_as_text(self, vocab_path):
        """Materializes the vocabulary in a human readable format. Vocabulary is a 
        dict of token_id to bytes (encoded using utf-8). It will be shown as
        a list of pairs (token_id, text (decoded using utf-8))
        """
        with open(vocab_path, "w", encoding="utf-8") as file:
            for key, val in self._vocabulary.items():
                try:
                    file.write(str(key) + "," + val.decode("utf-8") + "\n")
                except UnicodeDecodeError:
                    continue # Skip the bad byte
        print("Vocab materialized text data: " + str(vocab_path))

    def materialize_merges_as_text(self, merges_path):
        """Materializes the merges in a human readable format. Every row will be"
         a pair of text that got merged."""
        with open(merges_path, "w", encoding="utf-8") as file:
            for m in self._bpe_merges:
                try:
                    file.write(m[0].decode("utf-8") + "," + m[1].decode("utf-8") + "\n")
                except UnicodeDecodeError:
                    continue # Skip the bad byte
        print("Merges materialized text data: " + str(merges_path))


    def print_debug_string(self):
        """Helper function to print various variables to help with debugging."""
        print("Vocabulary size: " + str(len(self._vocabulary)))
        print("Frequency count table: " + str(len(self._frequency_count)))
        return
