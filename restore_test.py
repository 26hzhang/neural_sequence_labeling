from models.model import SeqLabelModel
from models.config import Config
from utils import Dataset, Processor, interactive_shell
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings


def main():
    # load configurations
    config = Config()

    re_train = False

    # create word and tag processor
    word_processor = Processor(config.word_vocab_filename, config.char_vocab_filename, lowercase=True, use_chars=True,
                               allow_unk=True)
    tag_processor = Processor(config.tag_filename)

    # load test dataset
    train_set = Dataset(config.train_filename, config.tag_idx, word_processor, tag_processor, max_iter=config.max_iter)
    dev_set = Dataset(config.dev_filename, config.tag_idx, word_processor, tag_processor, max_iter=config.max_iter)
    test_set = Dataset(config.test_filename, config.tag_idx, word_processor, tag_processor, max_iter=config.max_iter)

    # build model
    model = SeqLabelModel(config)
    model.restore_last_session(ckpt_path='ckpt/{}/'.format(config.train_task))

    # train
    if re_train:
        model.train(train_set, dev_set, test_set)

    # test
    model.evaluate(test_set, eval_dev=False)
    # interact
    idx_to_tag = {idx: tag for tag, idx in config.tag_vocab.items()}
    interactive_shell(model, word_processor, idx_to_tag)


if __name__ == '__main__':
    main()
