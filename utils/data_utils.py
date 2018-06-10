import ujson
import codecs
import random


def load_dataset(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        dataset = ujson.load(f)
    return dataset


def pad_sequences(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        # 0: "PAD" for words and chars, "O" for tags
        pad_tok = 0
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_sequences(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_sequences(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_sequences(sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length)
    sequence_length, _ = pad_sequences(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def process_batch_data(batch_words, batch_chars, batch_tags=None):
    b_words, b_words_len = pad_sequences(batch_words)
    b_chars, b_chars_len = pad_char_sequences(batch_chars)
    if batch_tags is None:
        return {"words": b_words, "chars": b_chars, "seq_len": b_words_len, "char_seq_len": b_chars_len,
                "batch_size": len(b_words)}
    else:
        b_tags, _ = pad_sequences(batch_tags)
        return {"words": b_words, "chars": b_chars, "tags": b_tags, "seq_len": b_words_len, "char_seq_len": b_chars_len,
                "batch_size": len(b_words)}


def dataset_batch_iter(dataset, batch_size):
    batch_words, batch_chars, batch_tags = [], [], []
    for record in dataset:
        batch_words.append(record["words"])
        batch_chars.append(record["chars"])
        batch_tags.append(record["tags"])
        if len(batch_words) == batch_size:
            yield process_batch_data(batch_words, batch_chars, batch_tags)
            batch_words, batch_chars, batch_tags = [], [], []
    if len(batch_words) > 0:
        yield process_batch_data(batch_words, batch_chars, batch_tags)


def batchnize_dataset(data, batch_size=None, shuffle=True):
    if type(data) == str:
        dataset = load_dataset(data)
    else:
        dataset = data
    if shuffle:
        random.shuffle(dataset)
    batches = []
    if batch_size is None:
        for batch in dataset_batch_iter(dataset, len(dataset)):
            batches.append(batch)
        return batches[0]
    else:
        for batch in dataset_batch_iter(dataset, batch_size):
            batches.append(batch)
        return batches


def align_data(data):
    """Given dict with lists, creates aligned strings
    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]
    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                             data_align["y"] = "O O    O  "
    """
    spacings = [max([len(seq[i]) for seq in data.values()]) for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()
    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ''
        for token, spacing in zip(seq, spacings):
            str_aligned += token + ' ' * (spacing - len(token) + 1)
        data_aligned[key] = str_aligned
    return data_aligned
