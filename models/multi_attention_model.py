import random
import tensorflow as tf
from utils import Progbar, batchnize_dataset
from models import BaseModel, add_timing_signal, multi_conv1d, layer_normalize, multi_head_attention
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, DropoutWrapper, ResidualWrapper
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


class SequenceLabelModel(BaseModel):
    def __init__(self, config):
        super(SequenceLabelModel, self).__init__(config)

    def _add_placeholders(self):
        self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")
        self.tags = tf.placeholder(tf.int32, shape=[None, None], name="tags")
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        if self.cfg["use_chars"]:
            self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name="chars")
            self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        # hyper-parameters
        self.emb_drop_rate = tf.placeholder(tf.float32, name="emb_drop_rate")
        self.rnn_keep_prob = tf.placeholder(tf.float32, name="rnn_keep_prob")
        self.attn_drop_rate = tf.placeholder(tf.float32, name="attn_drop_rate")  # drop_rate = 1.0 - keep_prob
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

    def _get_feed_dict(self, batch, emb_keep_prob=1.0, rnn_keep_prob=1.0, attn_keep_prob=1.0, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
        if "tags" in batch:
            feed_dict[self.tags] = batch["tags"]
        if self.cfg["use_chars"]:
            feed_dict[self.chars] = batch["chars"]
            feed_dict[self.char_seq_len] = batch["char_seq_len"]
        feed_dict[self.emb_drop_rate] = 1.0 - emb_keep_prob
        feed_dict[self.rnn_keep_prob] = rnn_keep_prob
        feed_dict[self.attn_drop_rate] = 1.0 - attn_keep_prob
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _create_single_rnn_cell(self, num_units):
        cell = GRUCell(num_units) if self.cfg["cell_type"] == "gru" else LSTMCell(num_units)
        if self.cfg["use_dropout"]:
            cell = DropoutWrapper(cell, output_keep_prob=self.rnn_keep_prob)
        if self.cfg["use_residual"]:
            cell = ResidualWrapper(cell)
        return cell

    def _build_embedding_op(self):
        with tf.variable_scope("embeddings"):
            pad_emb = tf.Variable(tf.zeros([1, self.cfg["emb_dim"]], dtype=tf.float32), name='pad_emb', trainable=False)
            emb = tf.get_variable(name="emb", dtype=tf.float32, shape=[self.word_vocab_size - 1, self.cfg["emb_dim"]],
                                  trainable=True)
            self.embeddings = tf.concat([pad_emb, emb], axis=0)
            word_emb = tf.nn.embedding_lookup(self.embeddings, self.words, name="word_emb")
            if self.cfg["add_positional_emb"]:
                word_emb = add_timing_signal(word_emb)
            self.word_emb = tf.layers.dropout(word_emb, rate=self.emb_drop_rate, training=self.is_train)
            if self.cfg["use_chars"]:
                c_pad_emb = tf.Variable(tf.zeros([1, self.cfg["char_emb_dim"]], dtype=tf.float32), name="c_pad_emb",
                                        trainable=False)
                c_emb = tf.get_variable(name="c_emb", dtype=tf.float32, trainable=True,
                                        shape=[self.char_vocab_size - 1, self.cfg["char_emb_dim"]])
                self.char_embeddings = tf.concat([c_pad_emb, c_emb], axis=0)
                char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.chars, name="char_emb")
                char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
                                              drop_rate=self.emb_drop_rate, is_train=self.is_train)
                if self.cfg["add_positional_emb"]:
                    char_represent = add_timing_signal(char_represent)
                self.chars_emb = tf.layers.dropout(char_represent, rate=self.emb_drop_rate, training=self.is_train)
                print("chars embeddings shape: {}".format(self.chars_emb.get_shape().as_list()))
            print("word embeddings shape: {}".format(self.word_emb.get_shape().as_list()))

    def _build_model_op(self):
        with tf.variable_scope("bi_directional_rnn"):
            cell_fw = self._create_single_rnn_cell(self.cfg["num_units"])
            cell_bw = self._create_single_rnn_cell(self.cfg["num_units"])
            if self.cfg["use_residual"]:
                self.word_emb = tf.layers.dense(self.word_emb, units=self.cfg["num_units"], use_bias=False,
                                                name="word_input_project")
                if self.cfg["use_chars"]:
                    self.chars_emb = tf.layers.dense(self.chars_emb, units=self.cfg["num_units"], use_bias=False,
                                                     name="chars_input_project")

            rnn_outs, _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_emb, sequence_length=self.seq_len,
                                                    dtype=tf.float32, scope="bi_rnn")
            rnn_outs = tf.concat(rnn_outs, axis=-1)
            print("Bi-directional RNN output shape on word: {}".format(rnn_outs.get_shape().as_list()))
            if self.cfg["use_chars"]:
                tf.get_variable_scope().reuse_variables()
                chars_rnn_outs, _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, self.chars_emb, dtype=tf.float32,
                                                              sequence_length=self.seq_len, scope="bi_rnn")
                chars_rnn_outs = tf.concat(chars_rnn_outs, axis=-1)
                print("Bi-directional RNN output shape on chars: {}".format(chars_rnn_outs.get_shape().as_list()))
                rnn_outs = rnn_outs + chars_rnn_outs
            rnn_outs = layer_normalize(rnn_outs)

        with tf.variable_scope("multi_head_attention"):
            attn_outs = multi_head_attention(rnn_outs, rnn_outs, self.cfg["num_heads"], self.cfg["attention_size"],
                                             drop_rate=self.attn_drop_rate, is_train=self.is_train)
            if self.cfg["use_residual"]:
                attn_outs = attn_outs + rnn_outs
            attn_outs = layer_normalize(attn_outs)  # residual connection and layer norm
            print("multi-heads attention output shape: {}".format(attn_outs.get_shape().as_list()))

        with tf.variable_scope("projection"):
            self.logits = tf.layers.dense(attn_outs, units=self.tag_vocab_size, use_bias=True)
            print("logits shape: {}".format(self.logits.get_shape().as_list()))

    def train_epoch(self, train_set, valid_data, epoch, shuffle=True):
        if shuffle:
            random.shuffle(train_set)
        train_set = batchnize_dataset(train_set, self.cfg.batch_size)
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        for i, batch_data in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch_data, emb_keep_prob=self.cfg["emb_keep_prob"],
                                            rnn_keep_prob=self.cfg["rnn_keep_prob"],
                                            attn_keep_prob=self.cfg["attn_keep_prob"], is_train=True, lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            self.train_writer.add_summary(summary, cur_step)
            if i % 100 == 0:
                valid_feed_dict = self._get_feed_dict(valid_data)
                valid_summary = self.sess.run(self.summary, feed_dict=valid_feed_dict)
                self.test_writer.add_summary(valid_summary, cur_step)
