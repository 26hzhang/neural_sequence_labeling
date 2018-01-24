import os
from models.data_process import load_vocab, load_glove_embeddings
from models.logger import get_logger


class Config(object):
    def __init__(self):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.logger = get_logger(self.log_file)
        self.word_vocab = load_vocab(self.word_vocab_filename)
        self.char_vocab = load_vocab(self.char_vocab_filename)
        self.tag_vocab = load_vocab(self.tag_filename)
        self.word_vocab_size = len(self.word_vocab)
        self.char_vocab_size = len(self.char_vocab)
        self.tag_vocab_size = len(self.tag_vocab)
        self.glove_embeddings = load_glove_embeddings(self.embedding_filename)

    """train task"""
    train_task = 'ner'  # pos, chunk, ner
    tag_idx = 3 if train_task == 'ner' else 2 if train_task == 'chunk' else 1

    """file path"""
    __head_dir = 'data/conll2003/en/raw/'
    train_filename = __head_dir + 'train.txt'
    dev_filename = __head_dir + 'valid.txt'
    test_filename = __head_dir + 'test.txt'
    word_vocab_filename = __head_dir + 'words.txt'
    char_vocab_filename = __head_dir + 'chars.txt'
    tag_filename = __head_dir + '{}_tags.txt'.format(train_task)

    """checkpoint and log files"""
    ckpt_path = 'ckpt/{}/'.format(train_task)  # path to save trained model
    log_file = ckpt_path + 'log_{}.txt'.format(train_task)  # log file
    model_name = 'seq_labeling_{}'.format(train_task)

    """set sentence and word maximal length"""
    max_len_sent = None  # if None, use the max length of sentence in each batch (changeable)
    max_len_word = None  # if None, use the max length of word in each batch (changeable)

    """word embeddings"""
    use_pretrained = True  # use pretrained word embeddings (GloVe embeddings)
    finetune_emb = False
    word_dim = 300  # word embeddings dim, 50, 100, 200, 300, default 300
    embedding_filename = __head_dir + 'glove.6B.{}d.filtered.npz'.format(word_dim)

    """char embeddings"""
    use_char_emb = True  # use char embeddings
    char_dim = 100  # char embeddings dimension, default 100
    char_rep_method = 'cnn'  # rnn, cnn
    char_out_size = 200
    # RNN hidden neuron size for char representation
    num_units_char = 100  # char hidden size should equal to char_represent_size/2
    # CNN filter size and height for char representation
    filter_sizes = [100, 100]  # sum of filter sizes should equal to char_represent_size
    heights = [5, 5]

    """training parameters"""
    epochs = 30  # number of epochs
    keep_prob = 0.5  # dropout keep probability
    grad_clip = 5.0  # positive value, if None, no clipping, default None
    batch_size = 20  # batch size for training
    max_iter = None  # set max iterations, if None, it equals number of batch in dataset
    max_to_keep = 5  # max session to save while training
    no_imprv_threshold = 5  # performs early stopping

    """learning method, rate and decay"""
    lr_method = 'adam'  # adam, nadam, sgd, adagrad, rmsprop, adadelta
    lr = 0.001  # initial learning rate
    lr_decay_method = 2  # 1: using lr_decay, 2 or otherwise: lr_decay_rate
    lr_decay = 0.9  # lr = lr * lr_decay for each epoch
    lr_decay_rate = 0.05  # lr = lr_init / (1 + lr_decay_rate * epoch) for each epoch

    """highway networks"""
    use_highway = True
    highway_num_layers = 2

    """model parameters"""
    mode_type = 'single'  # single, stack, complex
    num_layers = 2  # used for stack mode
    cell_type = 'lstm'  # lstm, gru
    num_units = 300  # LSTM hidden neuron size for labeling model
    use_crf = True  # use CRF
