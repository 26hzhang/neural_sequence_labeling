import tensorflow as tf
import os
from data.media_prepro import process_data
from models.multi_attention_model import SequenceLabelModel
from utils import batchnize_dataset, load_dataset

# dataset parameters
tf.flags.DEFINE_string("raw_path", "data/raw/media", "path to raw dataset")
tf.flags.DEFINE_string("save_path", "data/dataset/media", "path to save dataset")
tf.flags.DEFINE_string("language", "french", "language")

# dataset for train, validate and test
tf.flags.DEFINE_string("vocab", "data/dataset/media/vocab.json", "path to the word and tag vocabularies")
tf.flags.DEFINE_string("train_set", "data/dataset/media/train.json", "path to the training datasets")
tf.flags.DEFINE_string("dev_set", "data/dataset/media/dev.json", "path to the development datasets")
tf.flags.DEFINE_string("test_set", "data/dataset/media/test.json", "path to the test datasets")

# network parameters
tf.flags.DEFINE_string("cell_type", "lstm", "RNN cell for encoder and decoder: [lstm | gru], default: lstm")
tf.flags.DEFINE_integer("num_units", 128, "number of hidden units in each layer")
tf.flags.DEFINE_integer("num_layers", 2, "number of layers for rnns")
tf.flags.DEFINE_integer("emb_dim", 200, "embedding dimension for encoder and decoder input words/tokens")
tf.flags.DEFINE_boolean("use_dropout", True, "use dropout for rnn cells")
tf.flags.DEFINE_boolean("use_residual", True, "use residual connection for rnn cells")
tf.flags.DEFINE_integer("attention_size", None, "attention size for multi-head attention mechanism")
tf.flags.DEFINE_integer("num_heads", 8, "number of heads")
tf.flags.DEFINE_boolean("use_chars", True, "use char embeddings")
tf.flags.DEFINE_integer("char_emb_dim", 50, "character embedding dimension")
tf.flags.DEFINE_multi_integer("filter_sizes", [25, 25, 25, 25], "filter size")
tf.flags.DEFINE_multi_integer("channel_sizes", [5, 5, 5, 5], "channel size")
tf.flags.DEFINE_boolean("add_positional_emb", False, "add positional embedding")
tf.flags.DEFINE_boolean("use_crf", True, "use CRF decode")

# training parameters
tf.flags.DEFINE_float("lr", 0.001, "learning rate")
tf.flags.DEFINE_string("optimizer", "adam", "Optimizer: [adagrad | sgd | rmsprop | adadelta | adam], default: adam")
tf.flags.DEFINE_boolean("use_lr_decay", True, "apply learning rate decay for each epoch")
tf.flags.DEFINE_float("lr_decay", 0.9, "learning rate decay factor")
tf.flags.DEFINE_float("minimal_lr", 1e-4, "minimal learning rate")
tf.flags.DEFINE_float("grad_clip", 1.0, "maximal gradient norm")
tf.flags.DEFINE_float("emb_keep_prob", 0.7, "dropout keep probability for embedding while training")
tf.flags.DEFINE_float("rnn_keep_prob", 0.6, "dropout keep probability for RNN while training")
tf.flags.DEFINE_float("attn_keep_prob", 0.7, "dropout keep probability for attention while training")
tf.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.flags.DEFINE_integer("epochs", 100, "train epochs")
tf.flags.DEFINE_integer("max_to_keep", 5, "maximum trained models to be saved")
tf.flags.DEFINE_integer("no_imprv_tolerance", 5, "no improvement tolerance")
tf.flags.DEFINE_string("checkpoint_path", "ckpt/media/", "path to save models checkpoints")
tf.flags.DEFINE_string("summary_path", "ckpt/media/summary/", "path to save summaries")
tf.flags.DEFINE_string("model_name", "multi_attention_model", "models name")

# convert parameters to dict
config = tf.flags.FLAGS.flag_values_dict()

# create dataset from raw data files
if not os.path.exists(config["save_path"]) or not os.listdir(config["save_path"]):
    process_data(config)

print("Load datasets...")
train_data = load_dataset(config["train_set"])
valid_set = batchnize_dataset(config["dev_set"], config["batch_size"], shuffle=False)
test_set = batchnize_dataset(config["test_set"], config["batch_size"], shuffle=False)
valid_data = batchnize_dataset(config["dev_set"], shuffle=False)

print("Build models...")
model = SequenceLabelModel(config)
model.train(train_data, valid_data, valid_set, test_set)

print("Inference...")
sentences = ["alors une nuit le DIZAINE MOIS UNITE MILLE UNITE", "dans un hôtel à XVILLE dans le centre ville"]
ground_truths = ["O B-sejour-nbNuit I-sejour-nbNuit B-temps-date I-temps-date I-temps-date B-temps-annee "
                 "I-temps-annee I-temps-annee",
                 "B-objetBD I-objetBD I-objetBD B-localisation-ville I-localisation-ville "
                 "B-localisation-lieuRelatif-general I-localisation-lieuRelatif-general "
                 "I-localisation-lieuRelatif-general I-localisation-lieuRelatif-general"]
for sentence, truth in zip(sentences, ground_truths):
    result = model.inference(sentence)
    print(result)
    print("Ground truth:\n{}\n".format(truth))
