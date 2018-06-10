import os
from tqdm import tqdm
from collections import Counter
import numpy as np
import codecs
import re
from data.common import SPACE, UNK, PAD, NUM, END, write_json

# pre-set number of records in different glove embeddings
glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}

# Comma, period & question mark only:
PUNCTUATION_VOCABULARY = [SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK"]
PUNCTUATION_MAPPING = {"!EXCLAMATIONMARK": ".PERIOD", ":COLON": ",COMMA", ";SEMICOLON": ".PERIOD", "-DASH": ",COMMA"}

EOS_TOKENS = {".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"}
# punctuations that are not included in vocabulary nor mapping, must be added to CRAP_TOKENS
CRAP_TOKENS = {"<assets>", "<assets.>"}


def is_number(word):
    numbers = re.compile(r"\d")
    return len(numbers.sub("", word)) / len(word) < 0.6


def build_vocab_list(data_files, min_word_count, min_char_count, max_vocab_size):
    word_counter = Counter()
    char_counter = Counter()
    for file in data_files:
        with codecs.open(file, mode="r", encoding="utf-8") as f:
            for line in f:
                for word in line.lstrip().rstrip().split():
                    if word in CRAP_TOKENS or word in PUNCTUATION_VOCABULARY or word in PUNCTUATION_MAPPING:
                        continue
                    if is_number(word):
                        word_counter[NUM] += 1
                        for char in word:
                            char_counter[char] += 1
                        continue
                    word_counter[word] += 1
                    for char in word:
                        char_counter[char] += 1
    word_vocab = [word for word, count in word_counter.most_common() if count >= min_word_count and word != UNK and
                  word != NUM][:max_vocab_size]
    char_vocab = [char for char, count in char_counter.most_common() if count >= min_char_count and char != UNK]
    return word_vocab, char_vocab


def build_vocabulary(word_vocab, char_vocab):
    if NUM not in word_vocab:
        word_vocab.append(NUM)
    if END not in word_vocab:
        word_vocab.append(END)
    if UNK not in word_vocab:
        word_vocab.append(UNK)
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    if END not in char_vocab:
        char_vocab.append(END)
    if UNK not in char_vocab:
        char_vocab.append(UNK)
    if PAD not in char_vocab:
        char_vocab = [PAD] + char_vocab
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return word_dict, char_dict


def load_glove_vocab(glove_path, glove_name):
    vocab = set()
    total = glove_sizes[glove_name]
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            vocab.add(line[0])
    return vocab


def filter_glove_emb(word_dict, glove_path, glove_name, dim):
    scale = np.sqrt(3.0 / dim)
    vectors = np.random.uniform(-scale, scale, [len(word_dict), dim])
    mask = np.zeros([len(word_dict)])
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=glove_sizes[glove_name], desc="Filter glove embeddings"):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = [float(x) for x in line[1:]]
            if word in word_dict:
                word_idx = word_dict[word]
                mask[word_idx] = 1
                vectors[word_idx] = np.asarray(vector)
            # since tokens in train sets are lowercase
            elif word.lower() in word_dict and mask[word_dict[word.lower()]] == 0:
                word = word.lower()
                word_idx = word_dict[word]
                mask[word_idx] = 1
                vectors[word_idx] = np.asarray(vector)
    return vectors


def build_dataset(data_files, word_dict, char_dict, punct_dict, max_sequence_len):
    """
    data will consist of two sets of aligned sub-sequences (words and punctuations) of MAX_SEQUENCE_LEN tokens
    (actually punctuation sequence will be 1 element shorter).
    If a sentence is cut, then it will be added to next subsequence entirely
    (words before the cut belong to both sequences)
    """
    dataset = []
    current_words, current_chars, current_punctuations = [], [], []
    last_eos_idx = 0  # if it's still 0 when MAX_SEQUENCE_LEN is reached, then the sentence is too long and skipped.
    last_token_was_punctuation = True  # skip first token if it's punctuation
    # if a sentence does not fit into subsequence, then we need to skip tokens until we find a new sentence
    skip_until_eos = False
    for file in data_files:
        with codecs.open(file, 'r', encoding='utf-8') as f:
            for line in f:
                for token in line.split():
                    # First map oov punctuations to known punctuations
                    if token in PUNCTUATION_MAPPING:
                        token = PUNCTUATION_MAPPING[token]
                    if skip_until_eos:
                        if token in EOS_TOKENS:
                            skip_until_eos = False
                        continue
                    elif token in CRAP_TOKENS:
                        continue
                    elif token in punct_dict:
                        # if we encounter sequences like: "... !EXLAMATIONMARK .PERIOD ...",
                        # then we only use the first punctuation and skip the ones that follow
                        if last_token_was_punctuation:
                            continue
                        if token in EOS_TOKENS:
                            last_eos_idx = len(current_punctuations)  # no -1, because the token is not added yet
                        punctuation = punct_dict[token]
                        current_punctuations.append(punctuation)
                        last_token_was_punctuation = True
                    else:
                        if not last_token_was_punctuation:
                            current_punctuations.append(punct_dict[SPACE])
                        chars = []
                        for c in token:
                            c = char_dict.get(c, char_dict[UNK])
                            chars.append(c)
                        if is_number(token):
                            token = NUM
                        word = word_dict.get(token, word_dict[UNK])
                        current_words.append(word)
                        current_chars.append(chars)
                        last_token_was_punctuation = False
                    if len(current_words) == max_sequence_len:  # this also means, that last token was a word
                        assert len(current_words) == len(current_punctuations) + 1, \
                            "#words: %d; #punctuations: %d" % (len(current_words), len(current_punctuations))
                        # Sentence did not fit into subsequence - skip it
                        if last_eos_idx == 0:
                            skip_until_eos = True
                            current_words = []
                            current_chars = []
                            current_punctuations = []
                            # next sequence starts with a new sentence, so is preceded by eos which is punctuation
                            last_token_was_punctuation = True
                        else:
                            subsequence = {"words": current_words[:-1] + [word_dict[END]],
                                           "chars": current_chars[:-1] + [[char_dict[END]]],
                                           "tags": current_punctuations}
                            dataset.append(subsequence)
                            # Carry unfinished sentence to next subsequence
                            current_words = current_words[last_eos_idx + 1:]
                            current_chars = current_chars[last_eos_idx + 1:]
                            current_punctuations = current_punctuations[last_eos_idx + 1:]
                        last_eos_idx = 0  # sequence always starts with a new sentence
    return dataset


def process_data(config):
    train_file = os.path.join(config["raw_path"], "train.txt")
    dev_file = os.path.join(config["raw_path"], "dev.txt")
    ref_file = os.path.join(config["raw_path"], "ref.txt")
    asr_file = os.path.join(config["raw_path"], "asr.txt")
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    # build vocabulary
    word_vocab, char_vocab = build_vocab_list([train_file], config["min_word_count"], config["min_char_count"],
                                              config["max_vocab_size"])
    if not config["use_pretrained"]:
        word_dict, char_dict = build_vocabulary(word_vocab, char_vocab)
    else:
        glove_path = config["glove_path"].format(config["glove_name"], config["emb_dim"])
        glove_vocab = load_glove_vocab(glove_path, config["glove_name"])
        glove_vocab = glove_vocab & {word.lower() for word in glove_vocab}
        word_vocab = [word for word in word_vocab if word in glove_vocab]
        word_dict, char_dict = build_vocabulary(word_vocab, char_vocab)
        tmp_word_dict = word_dict.copy()
        del tmp_word_dict[UNK], tmp_word_dict[NUM], tmp_word_dict[END]
        vectors = filter_glove_emb(tmp_word_dict, glove_path, config["glove_name"], config["emb_dim"])
        np.savez_compressed(config["pretrained_emb"], embeddings=vectors)
    # create indices dataset
    punct_dict = dict([(punct, idx) for idx, punct in enumerate(PUNCTUATION_VOCABULARY)])
    train_set = build_dataset([train_file], word_dict, char_dict, punct_dict, config["max_sequence_len"])
    dev_set = build_dataset([dev_file], word_dict, char_dict, punct_dict, config["max_sequence_len"])
    ref_set = build_dataset([ref_file], word_dict, char_dict, punct_dict, config["max_sequence_len"])
    asr_set = build_dataset([asr_file], word_dict, char_dict, punct_dict, config["max_sequence_len"])
    vocab = {"word_dict": word_dict, "char_dict": char_dict, "tag_dict": punct_dict}
    # write to file
    write_json(config["vocab"], vocab)
    write_json(config["train_set"], train_set)
    write_json(config["dev_set"], dev_set)
    write_json(config["ref_set"], ref_set)
    write_json(config["asr_set"], asr_set)
