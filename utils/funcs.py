import numpy as np

NONE = 'O'


def compute_accuracy_f1(ground_truth, predict_labels, seq_lengths, train_task, tag_vocab, ):
    accs, correct_preds, total_correct, total_preds = [], 0.0, 0.0, 0.0
    for labels, labels_pred, sequence_lengths in zip(ground_truth, predict_labels, seq_lengths):
        for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            accs += [a == b for (a, b) in zip(lab, lab_pred)]
            if train_task != 'pos':
                lab_chunks = set(get_chunks(lab, tag_vocab))
                lab_pred_chunks = set(get_chunks(lab_pred, tag_vocab))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)
    acc = f1 = np.mean(accs)  # if the train task is POS tagging, then do not compute f1 score, only accuracy
    if train_task != 'pos':
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return {"acc": 100 * acc, "f1": 100 * f1}


def batch_iter(dataset, batch_size):
    """Performs dataset iterator"""
    batch_x, batch_y = [], []
    for x, y in dataset:
        if len(batch_x) == batch_size:
            yield batch_x, batch_y
            batch_x, batch_y = [], []
        if type(x[0]) == tuple:
            x = zip(*x)
        batch_x += [x]
        batch_y += [y]
    if len(batch_x) != 0:
        yield batch_x, batch_y


def get_chunk_type(tok, idx_to_tag):
    """Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks


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


def interactive_shell(model, word_processor, idx_to_tag):
    """Creates interactive shell to play with model"""
    while True:
        sentence = input('input>\n')
        words_raw = sentence.strip().split(' ')
        if words_raw == ['exit']:
            break
        words = [word_processor.fit(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = model.predict([words])
        preds = [idx_to_tag[idx] for idx in list(pred_ids[0])]
        to_print = align_data({'input': words_raw, 'output': preds})
        for key, seq in to_print.items():
            model.logger.info(seq)
