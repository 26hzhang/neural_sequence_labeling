import tensorflow as tf
import numpy as np
from tensorflow.contrib.crf import viterbi_decode, crf_log_likelihood
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, MultiRNNCell
from utils import CoNLLeval, load_dataset, get_logger, process_batch_data, align_data
from data.common import word_convert, UNK
import os


class BaseModel:
    def __init__(self, config):
        self.cfg = config
        self._initialize_config()
        self.sess, self.saver = None, None
        self._add_placeholders()
        self._build_embedding_op()
        self._build_model_op()
        self._build_loss_op()
        self._build_train_op()
        print('params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        self.initialize_session()

    def _initialize_config(self):
        # create folders and logger
        if not os.path.exists(self.cfg["checkpoint_path"]):
            os.makedirs(self.cfg["checkpoint_path"])
        if not os.path.exists(self.cfg["summary_path"]):
            os.makedirs(self.cfg["summary_path"])
        self.logger = get_logger(os.path.join(self.cfg["checkpoint_path"], "log.txt"))
        # load dictionary
        dict_data = load_dataset(self.cfg["vocab"])
        self.word_dict, self.char_dict = dict_data["word_dict"], dict_data["char_dict"]
        self.tag_dict = dict_data["tag_dict"]
        del dict_data
        self.word_vocab_size = len(self.word_dict)
        self.char_vocab_size = len(self.char_dict)
        self.tag_vocab_size = len(self.tag_dict)
        self.rev_word_dict = dict([(idx, word) for word, idx in self.word_dict.items()])
        self.rev_char_dict = dict([(idx, char) for char, idx in self.char_dict.items()])
        self.rev_tag_dict = dict([(idx, tag) for tag, idx in self.tag_dict.items()])

    def initialize_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=self.cfg["max_to_keep"])
        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.cfg["checkpoint_path"])  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.cfg["checkpoint_path"] + self.cfg["model_name"], global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.cfg["summary_path"] + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.cfg["summary_path"] + "test")

    def reinitialize_weights(self, scope_name=None):
        """Reinitialize parameters in a scope"""
        if scope_name is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            variables = tf.contrib.framework.get_variables(scope_name)
            self.sess.run(tf.variables_initializer(variables))

    @staticmethod
    def variable_summaries(variable, name=None):
        with tf.name_scope(name or "summary"):
            mean = tf.reduce_mean(variable)
            tf.summary.scalar("mean", mean)  # add mean value
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
            tf.summary.scalar("stddev", stddev)  # add standard deviation value
            tf.summary.scalar("max", tf.reduce_max(variable))  # add maximal value
            tf.summary.scalar("min", tf.reduce_min(variable))  # add minimal value
            tf.summary.histogram("histogram", variable)  # add histogram

    @staticmethod
    def viterbi_decode(logits, trans_params, seq_len):
        viterbi_sequences = []
        for logit, lens in zip(logits, seq_len):
            logit = logit[:lens]  # keep only the valid steps
            viterbi_seq, viterbi_score = viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences

    def _create_single_rnn_cell(self, num_units):
        cell = GRUCell(num_units) if self.cfg["cell_type"] == "gru" else LSTMCell(num_units)
        return cell

    def _create_rnn_cell(self):
        if self.cfg["num_layers"] is None or self.cfg["num_layers"] <= 1:
            return self._create_single_rnn_cell(self.cfg["num_units"])
        else:
            MultiRNNCell([self._create_single_rnn_cell(self.cfg["num_units"]) for _ in range(self.cfg["num_layers"])])

    def _add_placeholders(self):
        raise NotImplementedError("To be implemented...")

    def _get_feed_dict(self, data):
        raise NotImplementedError("To be implemented...")

    def _build_embedding_op(self):
        raise NotImplementedError("To be implemented...")

    def _build_model_op(self):
        raise NotImplementedError("To be implemented...")

    def _build_loss_op(self):
        if self.cfg["use_crf"]:
            crf_loss, self.trans_params = crf_log_likelihood(self.logits, self.tags, self.seq_len)
            self.loss = tf.reduce_mean(-crf_loss)
        else:  # using softmax
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tags)
            mask = tf.sequence_mask(self.seq_len)
            self.loss = tf.reduce_mean(tf.boolean_mask(losses, mask))
        tf.summary.scalar("loss", self.loss)

    def _build_train_op(self):
        with tf.variable_scope("train_step"):
            if self.cfg["optimizer"] == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
            elif self.cfg["optimizer"] == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            elif self.cfg["optimizer"] == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            elif self.cfg["optimizer"] == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            else:  # default adam optimizer
                if self.cfg["optimizer"] != 'adam':
                    print('Unsupported optimizing method {}. Using default adam optimizer.'
                          .format(self.cfg["optimizer"]))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.cfg["grad_clip"] is not None and self.cfg["grad_clip"] > 0:
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, self.cfg["grad_clip"])
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = optimizer.minimize(self.loss)

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        if self.cfg["use_crf"]:
            logits, trans_params, seq_len = self.sess.run([self.logits, self.trans_params, self.seq_len],
                                                          feed_dict=feed_dict)
            return self.viterbi_decode(logits, trans_params, seq_len)
        else:
            pred_logits = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
            logits = self.sess.run(pred_logits, feed_dict=feed_dict)
            return logits

    def train_epoch(self, train_set, valid_data, epoch):
        raise NotImplementedError("To be implemented...")

    def train(self, train_set, valid_data, valid_set, test_set):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch, init_lr = -np.inf, 0, self.cfg["lr"]
        self._add_summary()
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info('Epoch {}/{}:'.format(epoch, self.cfg["epochs"]))
            self.train_epoch(train_set, valid_data, epoch)  # train epochs
            self.evaluate(valid_set, "dev")
            score = self.evaluate(test_set, "test")
            if self.cfg["use_lr_decay"]:  # learning rate decay
                self.cfg["lr"] = max(init_lr / (1.0 + self.cfg["lr_decay"] * epoch), self.cfg["minimal_lr"])
            if score["FB1"] > best_f1:
                best_f1 = score["FB1"]
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

    def evaluate(self, dataset, name):
        save_path = os.path.join(self.cfg["checkpoint_path"], "result.txt")
        predictions, groundtruth, words_list = list(), list(), list()
        for data in dataset:
            predicts = self._predict_op(data)
            for tags, preds, words, seq_len in zip(data["tags"], predicts, data["words"], data["seq_len"]):
                tags = [self.rev_tag_dict[x] for x in tags[:seq_len]]
                preds = [self.rev_tag_dict[x] for x in preds[:seq_len]]
                words = [self.rev_word_dict[x] for x in words[:seq_len]]
                predictions.append(preds)
                groundtruth.append(tags)
                words_list.append(words)
        ce = CoNLLeval()
        score = ce.conlleval(predictions, groundtruth, words_list, save_path)
        self.logger.info("{} dataset -- acc: {:04.2f}, pre: {:04.2f}, rec: {:04.2f}, FB1: {:04.2f}"
                         .format(name, score["accuracy"], score["precision"], score["recall"], score["FB1"]))
        return score

    def words_to_indices(self, words):
        """
        Convert input words into batchnized word/chars indices for inference
        :param words: input words
        :return: batchnized word indices
        """
        chars_idx = []
        for word in words:
            chars = [self.char_dict[char] if char in self.char_dict else self.char_dict[UNK] for char in word]
            chars_idx.append(chars)
        words = [word_convert(word, language=self.cfg["language"]) for word in words]
        words_idx = [self.word_dict[word] if word in self.word_dict else self.word_dict[UNK] for word in words]
        return process_batch_data([words_idx], [chars_idx])

    def inference(self, sentence):
        words = sentence.lstrip().rstrip().split(" ")
        data = self.words_to_indices(words)
        predicts = self._predict_op(data)
        predicts = [self.rev_tag_dict[idx] for idx in list(predicts[0])]
        results = align_data({"input": words, "output": predicts})
        return "{}\n{}".format(results["input"], results["output"])
