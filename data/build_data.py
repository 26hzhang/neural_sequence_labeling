import sys
import numpy as np
from models.data_process import UNK, NUM, PAD
import unicodedata
import re


def raw_data_iterator(filename, lowercase=False):
    with open(filename, 'r') as f:
        words, pos_tags, chunk_tags, ner_tags = [], [], [], []
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(words) != 0:
                    yield words, pos_tags, chunk_tags, ner_tags
                    words, pos_tags, chunk_tags, ner_tags = [], [], [], []
            else:
                word, pos, chunk, ner = line.split(' ')
                if lowercase:
                    word = word.lower()
                if is_digit(word):
                    word = NUM
                words += [word]
                pos_tags += [pos]
                chunk_tags += [chunk]
                ner_tags += [ner]


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


def build_vocabs(datasets):
    sys.stdout.write('Building dataset vocab...')
    vocab_words = set()
    vocab_pos = set()
    vocab_chunk = set()
    vocab_ner = set()
    for dataset in datasets:
        for words, pos_tags, chunk_tags, ner_tags in dataset:
            vocab_words.update(words)
            vocab_pos.update(pos_tags)
            vocab_chunk.update(chunk_tags)
            vocab_ner.update(ner_tags)
    sys.stdout.write(' done. Totally {} tokens.\n'.format(len(vocab_words)))
    return vocab_words, vocab_pos, vocab_chunk, vocab_ner


def build_char_vocab(datasets):
    sys.stdout.write('Building char vocab...')
    vocab_chars = set()
    for dataset in datasets:
        for words, *_ in dataset:
            for word in words:
                vocab_chars.update(word)
    sys.stdout.write(' done. Totally {} chars.\n'.format(len(vocab_chars)))
    return vocab_chars


def load_glove_vocab(filename):
    sys.stdout.write('Loading GloVe vocabs...')
    with open(filename, 'r') as f:
        vocab = {line.strip().split()[0] for line in f}
    sys.stdout.write(' done. Totally {} tokens.\n'.format(len(vocab)))
    return vocab


def filter_and_save_glove_vectors(vocab, glove_path, save_path, dim):
    sys.stdout.write('Filtering {} dim embeddings...'.format(dim))
    embeddings = np.zeros([len(vocab), dim])  # initialize by zeros
    scale = np.sqrt(3.0 / dim)
    embeddings[1:3] = np.random.uniform(-scale, scale, [2, dim])  # for NUM and UNK
    with open(glove_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    sys.stdout.write(' done. saving...')
    np.savez_compressed(save_path, embeddings=embeddings)
    sys.stdout.write(' done.\n')


def load_vocab(filename):
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx
    except IOError:
        raise "ERROR: Unable to locate file {}.".format(filename)
    return d


def write_vocab(vocab, filename):
    sys.stdout.write('Writing vocab...')
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
        sys.stdout.write(" done. {} tokens in {}\n".format(len(vocab), filename))


def main():
    data_head_dir = 'conll2003/en/raw/'
    embedding_dir = 'glove.6B/glove.6B.{}d.txt'
    # construct word, pos, chunk, ner vocabularies
    train = raw_data_iterator(data_head_dir + 'train.txt', lowercase=True)
    dev = raw_data_iterator(data_head_dir + 'valid.txt', lowercase=True)
    test = raw_data_iterator(data_head_dir + 'test.txt', lowercase=True)
    vocab_words, vocab_pos, vocab_chunk, vocab_ner = build_vocabs([train, dev, test])
    vocab_glove = load_glove_vocab(embedding_dir.format(50))
    vocab = vocab_words & vocab_glove  # distinct vocab
    vocab = [PAD, UNK, NUM] + list(vocab)  # add unknown token, number token and pad token

    write_vocab(vocab, data_head_dir + 'words.txt')
    write_vocab(vocab_pos, data_head_dir + 'pos_tags.txt')
    write_vocab(vocab_chunk, data_head_dir + 'chunk_tags.txt')
    write_vocab(vocab_ner, data_head_dir + 'ner_tags.txt')

    # filter glove embeddings
    vocab = load_vocab(data_head_dir + 'words.txt')
    glove_path = 'glove.6B.{}d.filtered.npz'
    filter_and_save_glove_vectors(vocab, embedding_dir.format(50), data_head_dir + glove_path.format(50), 50)
    filter_and_save_glove_vectors(vocab, embedding_dir.format(100), data_head_dir + glove_path.format(100), 100)
    filter_and_save_glove_vectors(vocab, embedding_dir.format(200), data_head_dir + glove_path.format(200), 200)
    filter_and_save_glove_vectors(vocab, embedding_dir.format(300), data_head_dir + glove_path.format(300), 300)

    # create chars vocabulary
    train = raw_data_iterator(data_head_dir + 'train.txt')
    dev = raw_data_iterator(data_head_dir + 'valid.txt')
    test = raw_data_iterator(data_head_dir + 'test.txt')
    vocab_chars = build_char_vocab([train, dev, test])
    vocab_chars = [PAD] + list(vocab_chars)
    write_vocab(vocab_chars, data_head_dir + 'chars.txt')


if __name__ == '__main__':
    main()
