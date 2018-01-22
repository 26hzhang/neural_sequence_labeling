from models.model import SeqLabelModel
from models.data_process import Dataset, Processor
from models.config import Config
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings


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


def main():
    # create configurations
    sys.stdout.write('load pre-defined configs and pre-processed dataset...')
    config = Config()
    # create word and tag processor
    word_processor = Processor(config.word_vocab_filename, config.char_vocab_filename, lowercase=True, use_chars=True,
                               allow_unk=True)
    tag_processor = Processor(config.tag_filename)
    # load train, development and test dataset
    train_set = Dataset(config.train_filename, config.tag_idx, word_processor, tag_processor, max_iter=config.max_iter)
    dev_set = Dataset(config.dev_filename, config.tag_idx, word_processor, tag_processor, max_iter=config.max_iter)
    test_set = Dataset(config.test_filename, config.tag_idx, word_processor, tag_processor, max_iter=config.max_iter)
    sys.stdout.write(' done.\n')
    # build model
    model = SeqLabelModel(config)
    model.train(train_set, dev_set, test_set)
    # testing
    model.evaluate(test_set, eval_dev=False)
    # interact
    idx_to_tag = {idx: tag for tag, idx in config.tag_vocab.items()}
    interactive_shell(model, word_processor, idx_to_tag)


if __name__ == '__main__':
    main()
