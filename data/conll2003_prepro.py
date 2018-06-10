import os
import codecs
import numpy as np
from tqdm import tqdm
from collections import Counter
from data.common import write_json, PAD, UNK, NUM, word_convert

glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}


def raw_dataset_iter(filename, task_name, keep_number, lowercase):
    with codecs.open(filename, mode="r", encoding="utf-8") as f:
        words, tags = [], []
        for line in f:
            line = line.lstrip().rstrip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):  # means read whole one sentence
                if len(words) != 0:
                    yield words, tags
                    words, tags = [], []
            else:
                word, pos, chunk, ner = line.split(" ")
                if task_name == "ner":
                    tag = ner
                elif task_name == "chunk":
                    tag = chunk
                else:
                    tag = pos
                word = word_convert(word, keep_number=keep_number, lowercase=lowercase)
                words.append(word)
                tags.append(tag)


def load_dataset(filename, task_name, keep_number=False, lowercase=True):
    dataset = []
    for words, tags in raw_dataset_iter(filename, task_name, keep_number, lowercase):
        dataset.append({"words": words, "tags": tags})
    return dataset


def load_glove_vocab(glove_path, glove_name):
    vocab = set()
    total = glove_sizes[glove_name]
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            vocab.add(line[0])
    return vocab


def build_word_vocab(datasets):
    word_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            words = record["words"]
            for word in words:
                word_counter[word] += 1
    word_vocab = [PAD, UNK, NUM] + [word for word, _ in word_counter.most_common(10000) if word != NUM]
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict


def build_tag_vocab(datasets, task_name):
    tag_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            tags = record["tags"]
            for tag in tags:
                tag_counter[tag] += 1
    if task_name == "ner":
        tag_vocab = [tag for tag, _ in tag_counter.most_common()]  # "O" acts as padding
        tag_dict = dict([(ner, idx) for idx, ner in enumerate(tag_vocab)])
    else:
        tag_vocab = [PAD] + [tag for tag, _ in tag_counter.most_common()]  # "<PAD>" acts as padding
        tag_dict = dict([(ner, idx) for idx, ner in enumerate(tag_vocab)])
    return tag_dict


def build_char_vocab(datasets):
    char_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            for word in record["words"]:
                for char in word:
                    char_counter[char] += 1
    word_vocab = [PAD, UNK] + [char for char, _ in char_counter.most_common()]
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict


def build_word_vocab_pretrained(datasets, glove_vocab):
    word_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            words = record["words"]
            for word in words:
                word_counter[word] += 1
    # build word dict
    word_vocab = [word for word, _ in word_counter.most_common() if word != NUM]
    word_vocab = [PAD, UNK, NUM] + list(set(word_vocab) & glove_vocab)
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict


def filter_glove_emb(word_dict, glove_path, glove_name, dim):
    # filter embeddings
    vectors = np.zeros([len(word_dict), dim])
    embeddings = np.zeros([len(word_dict), dim])  # initialize by zeros
    scale = np.sqrt(3.0 / dim)
    embeddings[1:3] = np.random.uniform(-scale, scale, [2, dim])  # for NUM and UNK
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=glove_sizes[glove_name], desc="Filter glove embeddings"):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = [float(x) for x in line[1:]]
            if word in word_dict:
                word_idx = word_dict[word]
                vectors[word_idx] = np.asarray(vector)
    return vectors


def build_dataset(data, word_dict, char_dict, tag_dict):
    dataset = []
    for record in data:
        chars_list = []
        words = []
        for word in record["words"]:
            chars = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            chars_list.append(chars)
            word = word_convert(word, keep_number=False, lowercase=True)
            words.append(word_dict[word] if word in word_dict else word_dict[UNK])
        tags = [tag_dict[tag] for tag in record["tags"]]
        dataset.append({"words": words, "chars": chars_list, "tags": tags})
    return dataset


def process_data(config):
    train_data = load_dataset(os.path.join(config["raw_path"], "train.txt"), config["task_name"])
    dev_data = load_dataset(os.path.join(config["raw_path"], "valid.txt"), config["task_name"])
    test_data = load_dataset(os.path.join(config["raw_path"], "test.txt"), config["task_name"])
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    # build vocabulary
    if not config["use_pretrained"]:
        word_dict = build_word_vocab([train_data, dev_data, test_data])
    else:
        glove_path = config["glove_path"].format(config["glove_name"], config["emb_dim"])
        glove_vocab = load_glove_vocab(glove_path, config["glove_name"])
        word_dict = build_word_vocab_pretrained([train_data, dev_data, test_data], glove_vocab)
        vectors = filter_glove_emb(word_dict, glove_path, config["glove_name"], config["emb_dim"])
        np.savez_compressed(config["pretrained_emb"], embeddings=vectors)
    tag_dict = build_tag_vocab([train_data, dev_data, test_data], config["task_name"])
    # build char dict
    train_data = load_dataset(os.path.join(config["raw_path"], "train.txt"), config["task_name"], keep_number=True,
                              lowercase=config["char_lowercase"])
    dev_data = load_dataset(os.path.join(config["raw_path"], "valid.txt"), config["task_name"], keep_number=True,
                            lowercase=config["char_lowercase"])
    test_data = load_dataset(os.path.join(config["raw_path"], "test.txt"), config["task_name"], keep_number=True,
                             lowercase=config["char_lowercase"])
    char_dict = build_char_vocab([train_data, dev_data, test_data])
    # create indices dataset
    train_set = build_dataset(train_data, word_dict, char_dict, tag_dict)
    dev_set = build_dataset(dev_data, word_dict, char_dict, tag_dict)
    test_set = build_dataset(test_data, word_dict, char_dict, tag_dict)
    vocab = {"word_dict": word_dict, "char_dict": char_dict, "tag_dict": tag_dict}
    # write to file
    write_json(os.path.join(config["save_path"], "vocab.json"), vocab)
    write_json(os.path.join(config["save_path"], "train.json"), train_set)
    write_json(os.path.join(config["save_path"], "dev.json"), dev_set)
    write_json(os.path.join(config["save_path"], "test.json"), test_set)
