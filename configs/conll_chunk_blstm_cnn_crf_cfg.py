import tensorflow as tf
import os
from data.conll2003_prepro import process_data


def create_configuration():
    # dataset parameters
    tf.flags.DEFINE_string("task_name", "chunk", "task name")
    tf.flags.DEFINE_string("language", "english", "language")  # used for inference, indicated the source language
    tf.flags.DEFINE_string("raw_path", "data/raw/conll2003/raw", "path to raw dataset")
    tf.flags.DEFINE_string("save_path", "data/dataset/conll2003/chunk", "path to save dataset")
    tf.flags.DEFINE_string("glove_name", "6B", "glove embedding name")
    tf.flags.DEFINE_boolean("char_lowercase", True, "char lowercase")
    # glove embedding path
    glove_path = os.path.join(os.path.expanduser('~'), "utilities", "embeddings", "glove.{}.{}d.txt")
    tf.flags.DEFINE_string("glove_path", glove_path, "glove embedding path")

    # dataset for train, validate and test
    tf.flags.DEFINE_string("vocab", "data/dataset/conll2003/chunk/vocab.json", "path to the word and tag vocabularies")
    tf.flags.DEFINE_string("train_set", "data/dataset/conll2003/chunk/train.json", "path to the training datasets")
    tf.flags.DEFINE_string("dev_set", "data/dataset/conll2003/chunk/dev.json", "path to the development datasets")
    tf.flags.DEFINE_string("test_set", "data/dataset/conll2003/chunk/test.json", "path to the test datasets")
    tf.flags.DEFINE_string("pretrained_emb", "data/dataset/conll2003/chunk/glove_emb.npz", "pretrained embeddings")

    # network parameters
    tf.flags.DEFINE_string("cell_type", "lstm", "RNN cell for encoder and decoder: [lstm | gru], default: lstm")
    tf.flags.DEFINE_integer("num_units", 300, "number of hidden units for rnn cell")
    tf.flags.DEFINE_integer("num_layers", None, "number of rnn layers")
    tf.flags.DEFINE_boolean("use_stack_rnn", False, "True: use stacked bi-rnn, False: use normal bi-rnn "
                                                    "(used for num_layers > 1)")
    tf.flags.DEFINE_boolean("use_pretrained", True, "use pretrained word embedding")
    tf.flags.DEFINE_boolean("tuning_emb", False, "tune pretrained word embedding while training")
    tf.flags.DEFINE_integer("emb_dim", 300, "embedding dimension for encoder and decoder input words/tokens")
    tf.flags.DEFINE_boolean("use_chars", True, "use char embeddings")
    tf.flags.DEFINE_boolean("use_residual", False, "use residual connection")
    tf.flags.DEFINE_boolean("use_layer_norm", False, "use layer normalization")
    tf.flags.DEFINE_integer("char_emb_dim", 100, "character embedding dimension")
    tf.flags.DEFINE_boolean("use_highway", True, "use highway network")
    tf.flags.DEFINE_integer("highway_layers", 2, "number of layers for highway network")
    tf.flags.DEFINE_integer("num_filters", 2, "number of filters for character representation")
    tf.flags.DEFINE_integer("filter_size", 100, "filter size")
    tf.flags.DEFINE_integer("channel_size", 5, "channel size")
    tf.flags.DEFINE_boolean("use_crf", True, "use CRF decoder")
    # attention mechanism (normal attention is Lurong/Bahdanau liked attention mechanism)
    tf.flags.DEFINE_string("use_attention", None, "use attention mechanism: [None | self_attention | normal_attention]")
    # Params for self attention (multi-head)
    tf.flags.DEFINE_integer("attention_size", None, "attention size for multi-head attention mechanism")
    tf.flags.DEFINE_integer("num_heads", 8, "number of heads")

    # training parameters
    tf.flags.DEFINE_float("lr", 0.001, "learning rate")
    tf.flags.DEFINE_string("optimizer", "adam", "optimizer: [adagrad | sgd | rmsprop | adadelta | adam], default: adam")
    tf.flags.DEFINE_boolean("use_lr_decay", True, "apply learning rate decay for each epoch")
    tf.flags.DEFINE_float("lr_decay", 0.05, "learning rate decay factor")
    tf.flags.DEFINE_float("minimal_lr", 1e-5, "minimal learning rate")
    tf.flags.DEFINE_float("grad_clip", 5.0, "maximal gradient norm")
    tf.flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability for embedding while training")
    tf.flags.DEFINE_integer("batch_size", 20, "batch size")
    tf.flags.DEFINE_integer("epochs", 100, "train epochs")
    tf.flags.DEFINE_integer("max_to_keep", 5, "maximum trained models to be saved")
    tf.flags.DEFINE_integer("no_imprv_tolerance", 5, "no improvement tolerance")
    tf.flags.DEFINE_string("checkpoint_path", "ckpt/conll2003_chunk/", "path to save models checkpoints")
    tf.flags.DEFINE_string("summary_path", "ckpt/conll2003_chunk/summary/", "path to save summaries")
    tf.flags.DEFINE_string("model_name", "chunk_blstm_cnn_crf_model", "models name")

    # convert parameters to dict
    tf_config = tf.flags.FLAGS.flag_values_dict()

    # create dataset from raw data files
    if not os.path.exists(tf_config["save_path"]) or not os.listdir(tf_config["save_path"]):
        process_data(tf_config)
    if not os.path.exists(tf_config["pretrained_emb"]) and tf_config["use_pretrained"]:
        process_data(tf_config)

    # return config
    return tf_config
