import os
import codecs
from collections import Counter
from data.common import write_json, PAD, UNK, word_convert


def raw_dataset_iter(filename, encoding="utf-8"):
    with codecs.open(filename, mode="r", encoding=encoding) as f:
        words, tags = [], []
        for line in f:
            line = line.lstrip().rstrip()
            if len(line) == 0 or line.startswith("--------------"):  # means read whole one sentence
                if len(words) != 0:
                    yield words, tags
                    words, tags = [], []
            else:
                _, word, tag = line.split("\t")
                word = word_convert(word, language="french")
                words.append(word)
                tags.append(tag)


def load_dataset(filename, encoding="utf-8"):
    dataset = []
    for words, tags in raw_dataset_iter(filename, encoding):
        dataset.append({"words": words, "tags": tags})
    return dataset


def build_vocab(datasets):
    char_counter = Counter()
    word_counter = Counter()
    tag_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            words = record["words"]
            for word in words:
                word_counter[word] += 1
                for char in word:
                    char_counter[char] += 1
            tags = record["tags"]
            for tag in tags:
                tag_counter[tag] += 1
    word_vocab = [PAD, UNK] + [word for word, _ in word_counter.most_common()]
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    char_vocab = [PAD, UNK] + [char for char, _ in char_counter.most_common()]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    tag_vocab = [tag for tag, _ in tag_counter.most_common()]
    tag_dict = dict([(tag, idx) for idx, tag in enumerate(tag_vocab)])
    return word_dict, char_dict, tag_dict


def build_dataset(data, word_dict, char_dict, tag_dict):
    dataset = []
    for record in data:
        chars_list = []
        for word in record["words"]:
            chars = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            chars_list.append(chars)
        words = [word_dict[word] if word in word_dict else word_dict[UNK] for word in record["words"]]
        tags = [tag_dict[tag] for tag in record["tags"]]
        dataset.append({"words": words, "chars": chars_list, "tags": tags})
    return dataset


def process_data(config):
    # load raw data
    train_data = load_dataset(os.path.join(config["raw_path"], "train.crf"), encoding="cp1252")
    dev_data = load_dataset(os.path.join(config["raw_path"], "dev.crf"), encoding="cp1252")
    test_data = load_dataset(os.path.join(config["raw_path"], "test.crf"), encoding="cp1252")
    # build vocabulary
    word_dict, char_dict, _ = build_vocab([train_data, dev_data])
    *_, tag_dict = build_vocab([train_data, dev_data, test_data])
    # create indices dataset
    train_set = build_dataset(train_data, word_dict, char_dict, tag_dict)
    dev_set = build_dataset(dev_data, word_dict, char_dict, tag_dict)
    test_set = build_dataset(test_data, word_dict, char_dict, tag_dict)
    vocab = {"word_dict": word_dict, "char_dict": char_dict, "tag_dict": tag_dict}
    # write to file
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    write_json(os.path.join(config["save_path"], "vocab.json"), vocab)
    write_json(os.path.join(config["save_path"], "train.json"), train_set)
    write_json(os.path.join(config["save_path"], "dev.json"), dev_set)
    write_json(os.path.join(config["save_path"], "test.json"), test_set)
