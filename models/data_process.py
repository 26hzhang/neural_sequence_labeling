import numpy as np
import unicodedata
import re

UNK = '<UNK>'
NUM = '<NUM>'
PAD = '<PAD>'  # default at the first place of embedding (padding zero)


class Dataset(object):
    def __init__(self, filename, tag_idx, word_processor=None, tag_processor=None, delimiter=' ', max_iter=None):
        self.filename = filename
        self.tag_idx = tag_idx
        self.word_processor = word_processor
        self.tag_processor = tag_processor
        self.delimiter = delimiter
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(self.delimiter)
                    word, tag = ls[0], ls[self.tag_idx]
                    if self.word_processor is not None:
                        word = self.word_processor.fit(word)
                    if self.tag_processor is not None:
                        tag = self.tag_processor.fit(tag)
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length


class Processor(object):
    """Transfer words and labels into corresponding ids"""
    def __init__(self, word_vocab_path, char_vocab_path=None, lowercase=False, use_chars=False, allow_unk=False):
        self.word_vocab = load_vocab(word_vocab_path)
        self.char_vocab = load_vocab(char_vocab_path) if char_vocab_path is not None else None
        self.lowercase = lowercase
        self.use_chars = use_chars
        self.allow_unk = allow_unk

    @staticmethod
    def is_digit(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        result = re.compile(r'^[-+]?[0-9]+,[0-9]+$').match(s)
        if result:
            return True
        return False

    def fit(self, word):
        char_ids = []
        # 0. get chars of words
        if self.char_vocab is not None and self.use_chars is True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in self.char_vocab:
                    char_ids += [self.char_vocab[char]]
        # 1. pre-process word
        if self.lowercase:
            word = word.lower()
        # if word.isdigit():
        if self.is_digit(word):
            word = NUM
        # 2. get id of word
        if self.word_vocab is not None:
            if word in self.word_vocab:
                word = self.word_vocab[word]
            else:
                if self.allow_unk:
                    word = self.word_vocab[UNK]
                else:
                    raise Exception("Unknown key is not allowed.")
        # 3. return tuple char ids, word id
        if self.char_vocab is not None and self.use_chars is True:
            return char_ids, word
        else:
            return word


def load_glove_embeddings(filename):
    """Load filtered GloVe vectors"""
    try:
        with np.load(filename) as data:
            return data["embeddings"]
    except IOError:
        raise "ERROR: Unable to locate file {}.".format(filename)


def load_vocab(filename):
    """Load vocabulary into dictionary"""
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx
    except IOError:
        raise "ERROR: Unable to locate file {}.".format(filename)
    return d


def _pad_sequences(sequences, pad_tok, max_length):
    """Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        if len(seq) < max_length:
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        else:
            seq_ = seq[:max_length]
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]
    return sequence_padded, sequence_length


def pad_sequences(sequences, max_length, pad_tok, max_length_word=None, nlevels=1):
    """Args:
        sequences: a generator of list or tuple
        max_length: maximal length for a sentence allowed
        max_length_word: maximal length for a word allow, only for nLevels=2
        pad_tok: the char to pad with
        nlevels: "depth" of padding, 2 for the case where we have characters ids
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []
    if nlevels == 1:
        if max_length is None:
            max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        if max_length_word is None:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]
        if max_length is None:
            max_length_sentence = max(map(lambda x: len(x), sequences))
        else:
            max_length_sentence = max_length
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    return sequence_padded, sequence_length
