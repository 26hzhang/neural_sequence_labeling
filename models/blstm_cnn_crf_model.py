import tensorflow as tf
import numpy as np
from models import BaseModel, AttentionCell, multi_head_attention
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from models.nns import multi_conv1d, highway_network, layer_normalize
from utils import Progbar


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
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.keep_prob = tf.placeholder(tf.float32, name="rnn_keep_probability")
        self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

    def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
        if "tags" in batch:
            feed_dict[self.tags] = batch["tags"]
        if self.cfg["use_chars"]:
            feed_dict[self.chars] = batch["chars"]
            feed_dict[self.char_seq_len] = batch["char_seq_len"]
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.drop_rate] = 1.0 - keep_prob
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _create_rnn_cell(self):
        if self.cfg["num_layers"] is None or self.cfg["num_layers"] <= 1:
            return self._create_single_rnn_cell(self.cfg["num_units"])
        else:
            if self.cfg["use_stack_rnn"]:
                return [self._create_single_rnn_cell(self.cfg["num_units"]) for _ in range(self.cfg["num_layers"])]
            else:
                return MultiRNNCell([self._create_single_rnn_cell(self.cfg["num_units"])
                                     for _ in range(self.cfg["num_layers"])])

    def _build_embedding_op(self):
        with tf.variable_scope("embeddings"):
            if not self.cfg["use_pretrained"]:
                self.word_embeddings = tf.get_variable(name="emb", dtype=tf.float32, trainable=True,
                                                       shape=[self.word_vocab_size, self.cfg["emb_dim"]])
            else:
                self.word_embeddings = tf.Variable(np.load(self.cfg["pretrained_emb"])["embeddings"], name="emb",
                                                   dtype=tf.float32, trainable=self.cfg["tuning_emb"])
            word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.words, name="word_emb")
            print("word embedding shape: {}".format(word_emb.get_shape().as_list()))
            if self.cfg["use_chars"]:
                self.char_embeddings = tf.get_variable(name="c_emb", dtype=tf.float32, trainable=True,
                                                       shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
                char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.chars, name="chars_emb")
                char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
                                              drop_rate=self.drop_rate, is_train=self.is_train)
                print("chars representation shape: {}".format(char_represent.get_shape().as_list()))
                word_emb = tf.concat([word_emb, char_represent], axis=-1)
            if self.cfg["use_highway"]:
                self.word_emb = highway_network(word_emb, self.cfg["highway_layers"], use_bias=True, bias_init=0.0,
                                                keep_prob=self.keep_prob, is_train=self.is_train)
            else:
                self.word_emb = tf.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
            print("word and chars concatenation shape: {}".format(self.word_emb.get_shape().as_list()))

    def _build_model_op(self):
        with tf.variable_scope("bi_directional_rnn"):
            cell_fw = self._create_rnn_cell()
            cell_bw = self._create_rnn_cell()
            if self.cfg["use_stack_rnn"]:
                rnn_outs, *_ = stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_emb, dtype=tf.float32,
                                                               sequence_length=self.seq_len)
            else:
                rnn_outs, *_ = bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_emb, sequence_length=self.seq_len,
                                                         dtype=tf.float32)
            rnn_outs = tf.concat(rnn_outs, axis=-1)
            rnn_outs = tf.layers.dropout(rnn_outs, rate=self.drop_rate, training=self.is_train)
            if self.cfg["use_residual"]:
                word_project = tf.layers.dense(self.word_emb, units=2 * self.cfg["num_units"], use_bias=False)
                rnn_outs = rnn_outs + word_project
            outputs = layer_normalize(rnn_outs) if self.cfg["use_layer_norm"] else rnn_outs
            print("rnn output shape: {}".format(outputs.get_shape().as_list()))

        if self.cfg["use_attention"] == "self_attention":
            with tf.variable_scope("self_attention"):
                attn_outs = multi_head_attention(outputs, outputs, self.cfg["num_heads"], self.cfg["attention_size"],
                                                 drop_rate=self.drop_rate, is_train=self.is_train)
                if self.cfg["use_residual"]:
                    attn_outs = attn_outs + outputs
                outputs = layer_normalize(attn_outs) if self.cfg["use_layer_norm"] else attn_outs
                print("self-attention output shape: {}".format(outputs.get_shape().as_list()))

        elif self.cfg["use_attention"] == "normal_attention":
            with tf.variable_scope("normal_attention"):
                context = tf.transpose(outputs, [1, 0, 2])
                p_context = tf.layers.dense(outputs, units=2 * self.cfg["num_units"], use_bias=False)
                p_context = tf.transpose(p_context, [1, 0, 2])
                attn_cell = AttentionCell(self.cfg["num_units"], context, p_context)  # time major based
                attn_outs, _ = dynamic_rnn(attn_cell, context, sequence_length=self.seq_len, time_major=True,
                                           dtype=tf.float32)
                outputs = tf.transpose(attn_outs, [1, 0, 2])
                print("attention output shape: {}".format(outputs.get_shape().as_list()))

        with tf.variable_scope("project"):
            self.logits = tf.layers.dense(outputs, units=self.tag_vocab_size, use_bias=True)
            print("logits shape: {}".format(self.logits.get_shape().as_list()))

    def train_epoch(self, train_set, valid_data, epoch):
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        for i, batch_data in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch_data, is_train=True, keep_prob=self.cfg["keep_prob"],
                                            lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            self.train_writer.add_summary(summary, cur_step)
            if i % 100 == 0:
                valid_feed_dict = self._get_feed_dict(valid_data)
                valid_summary = self.sess.run(self.summary, feed_dict=valid_feed_dict)
                self.test_writer.add_summary(valid_summary, cur_step)

    def train(self, train_set, valid_data, valid_set, test_set):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch, init_lr = -np.inf, 0, self.cfg["lr"]
        self._add_summary()
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info('Epoch {}/{}:'.format(epoch, self.cfg["epochs"]))
            self.train_epoch(train_set, valid_data, epoch)  # train epochs
            if self.cfg["use_lr_decay"]:  # learning rate decay
                self.cfg["lr"] = max(init_lr / (1.0 + self.cfg["lr_decay"] * epoch), self.cfg["minimal_lr"])
            if self.cfg["task_name"] == "pos":
                self.eval_accuracy(valid_set, "dev")
                acc = self.eval_accuracy(test_set, "test")
                cur_test_score = acc
            else:
                self.evaluate(valid_set, "dev")
                score = self.evaluate(test_set, "test")
                cur_test_score = score["FB1"]
            if cur_test_score > best_f1:
                best_f1 = cur_test_score
                no_imprv_epoch = 0
                self.save_session(epoch)
                self.logger.info(' -- new BEST score on test dataset: {:04.2f}'.format(best_f1))
            else:
                no_imprv_epoch += 1
                if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                    self.logger.info('early stop at {}th epoch without improvement, BEST score on testset: {:04.2f}'
                                     .format(epoch, best_f1))
                    break
        self.train_writer.close()
        self.test_writer.close()

    def eval_accuracy(self, dataset, name):  # Used for POS task
        accuracy = []
        for data in dataset:
            predicts = self._predict_op(data)
            for preds, tags, seq_len in zip(predicts, data["tags"], data["seq_len"]):
                preds = preds[:seq_len]
                tags = tags[:seq_len]
                accuracy += [p == t for p, t in zip(preds, tags)]
        acc = np.mean(accuracy) * 100.0
        self.logger.info("{} dataset -- accuracy: {:04.2f}".format(name, acc))
        return acc
