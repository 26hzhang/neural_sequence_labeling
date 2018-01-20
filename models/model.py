import tensorflow as tf
from models.base_model import BaseModel
from models.logger import Progbar
from models.preprocess import batch_iter, pad_sequences
from models.func import viterbi_decode, BiRNN, StackedBiRNN, dense, compute_accuracy_f1
import sys


class SeqLabelModel(BaseModel):
    def __init__(self, config):
        sys.stdout.write('Build model...')
        super(SeqLabelModel, self).__init__(config)
        self._add_placeholders()
        self._build_word_embeddings_op()
        self._build_model_op()
        self._build_pred_op()
        self._build_loss_op()
        self._build_train_op(self.config.learning_method, self.config.learning_rate, self.loss, self.config.grad_clip)
        self.initialize_session()
        sys.stdout.write(' done.\n')

    def _add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, self.config.max_len_sent], name="word_ids")
        # shape = (batch size)
        self.seq_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, self.config.max_len_sent, self.config.max_len_word],
                                       name="char_ids")
        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, self.config.max_len_sent], name="word_lengths")
        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, self.config.max_len_sent], name="labels")
        # hyper parameters
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

    def _get_feed_dict(self, words, labels=None, learning_rate=None, dropout_keep_prob=None):
        # perform padding of the given data
        if self.config.use_char_emb:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, self.config.max_len_sent, 0)
            char_ids, word_lengths = pad_sequences(char_ids, max_length=self.config.max_len_sent, pad_tok=0,
                                                   max_length_word=self.config.max_len_word, nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, max_length=self.config.max_len_sent, pad_tok=0)
            char_ids = word_lengths = None
        # build feed dictionary
        feed_dict = {self.word_ids: word_ids, self.seq_lengths: sequence_lengths}
        if self.config.use_char_emb:
            feed_dict[self.char_ids] = char_ids
            feed_dict[self.word_lengths] = word_lengths
        if labels is not None:
            labels, _ = pad_sequences(labels, max_length=self.config.max_len_sent, pad_tok=0)
            feed_dict[self.labels] = labels
        if learning_rate is not None:
            feed_dict[self.learning_rate] = learning_rate
        if dropout_keep_prob is not None:
            feed_dict[self.dropout_keep_prob] = dropout_keep_prob
        return feed_dict, sequence_lengths

    def _build_word_embeddings_op(self):
        with tf.variable_scope('words'):
            if self.config.use_pretrained:
                _word_embeddings = tf.Variable(self.config.glove_embeddings, name='_word_embeddings', dtype=tf.float32,
                                               trainable=self.config.train_embedding)
            else:
                _word_embeddings = tf.get_variable(name='_word_embeddings', dtype=tf.float32, trainable=True,
                                                   shape=[self.config.word_vocab_size, self.config.word_dim])
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")

        with tf.variable_scope('chars'):
            if self.config.use_char_emb:
                _char_embeddings = tf.get_variable(name='_char_embeddings', dtype=tf.float32, trainable=True,
                                                   shape=[self.config.char_vocab_size, self.config.char_dim])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="char_embeddings")
                s = tf.shape(char_embeddings)  # [batch size, max length of sentence, max length of word, char_dim]
                # put the time dimension on axis=1
                char_embeddings = tf.reshape(char_embeddings, shape=[s[0] * s[1], s[-2], self.config.char_dim])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])
                # bi-directional rnn to encode char embeddings
                char_bi_rnn = BiRNN(self.config.hidden_size_char, cell_type=self.config.cell_type, scope='char_bi_rnn')
                output = char_bi_rnn(char_embeddings, word_lengths, return_last_state=True)
                # shape = (batch size, max sentence length, char hidden size)
                self.char_output = tf.reshape(output, shape=[s[0], s[1], 2 * self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, self.char_output], axis=-1)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_keep_prob)  # dropout before bi-lstm

    def _build_model_op(self):
        with tf.variable_scope('bidirectional_rnn'):
            if self.config.stacked == 'simple_stack':
                # 2-layer stacked bidirectional lstm networks
                stacked_bi_rnn = StackedBiRNN(2, self.config.hidden_size_rnn, cell_type=self.config.cell_type,
                                              scope='simple_stack')
                output = stacked_bi_rnn(self.word_embeddings, self.seq_lengths, keep_prob=self.dropout_keep_prob)
            elif self.config.stacked == 'complex_stack':
                # first layer of bi-directional rnn
                merge_bi_rnn = BiRNN(self.config.hidden_size_rnn, cell_type=self.config.cell_type, scope='merge_bi_rnn')
                merge_output = merge_bi_rnn(self.word_embeddings, self.seq_lengths)
                # match char output shape to merge_output shape
                char_output = dense(self.char_output, 2 * self.config.hidden_size_rnn, scope='char_project')
                merge_output = tf.add(merge_output, char_output)
                # second layer of bi-directional rnn
                bi_rnn = BiRNN(self.config.hidden_size_rnn, cell_type=self.config.cell_type, scope='complex_stack')
                output = bi_rnn(merge_output, self.seq_lengths, keep_prob=self.dropout_keep_prob)
            else:  # default single model
                bi_rnn = BiRNN(self.config.hidden_size_rnn, cell_type=self.config.cell_type, scope='single_mode')
                output = bi_rnn(self.word_embeddings, self.seq_lengths, keep_prob=self.dropout_keep_prob)

        self.logits = dense(output, self.config.tag_vocab_size, use_bias=True, scope='project')

    def _build_loss_op(self):
        if self.config.use_crf:
            log_ll, self.trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.seq_lengths)
            self.loss = tf.reduce_mean(-log_ll)
        else:  # using softmax
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.seq_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

    def _build_pred_op(self):
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def train(self, train_set, dev_set, test_set):
        self.logger.info('Start training...')
        best_score = 0  # store the current best f1 score on dev_set, updated if new best one is derived
        no_imprv_epoch_count = 0  # count the continuous no improvement epochs
        num_batches = (len(train_set) + self.config.batch_size - 1) // self.config.batch_size  # number of batches
        init_learning_rate = self.config.learning_rate  # initial learning rate
        for epoch in range(1, self.config.epochs + 1):
            self.logger.info('Epoch %2d/%2d:' % (epoch, self.config.epochs))
            # run each epoch
            prog = Progbar(target=num_batches)
            for i, (words, labels) in enumerate(batch_iter(train_set, self.config.batch_size)):
                feed_dict, _ = self._get_feed_dict(words, labels, self.config.learning_rate,
                                                   self.config.dropout_keep_prob)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                prog.update(i + 1, [("train loss", train_loss)])
            self.evaluate(dev_set)  # evaluate dev_set
            metrics = self.evaluate(test_set, eval_dev=False)  # evaluate test_set
            cur_score = metrics['f1']
            # learning rate decay method
            if self.config.lr_decay_method == 1:
                self.config.learning_rate *= self.config.lr_decay
            else:
                self.config.learning_rate = init_learning_rate / (1 + self.config.lr_decay_rate * epoch)
            # performs early stop and parameters save
            if cur_score > best_score:
                no_imprv_epoch_count = 0
                self.save_session(epoch)  # save model with a new best score is obtained
                best_score = cur_score
                self.logger.info(' -- new best score: {}'.format(best_score))
            else:
                no_imprv_epoch_count += 1
                if no_imprv_epoch_count > self.config.no_imprv_threshold:
                    self.logger.info('early stop at %dth epoch without improvement for %d epochs, best score: %f' %
                                     (epoch, no_imprv_epoch_count, best_score))
                    # save the last one
                    self.save_session(epoch)
                    break
        self.logger.info('Training process done...')

    def predict(self, words):
        feed_dict, sequence_lengths = self._get_feed_dict(words, dropout_keep_prob=1.0)
        if self.config.use_crf:
            # get tag scores and transition params of CRF
            logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=feed_dict)
            return viterbi_decode(logits, trans_params, sequence_lengths), sequence_lengths
        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=feed_dict)
            return labels_pred, sequence_lengths

    def evaluate(self, dataset, eval_dev=True):
        if eval_dev:
            self.logger.info("Testing model over DEVELOPMENT dataset")
        else:
            self.logger.info('Testing model over TEST dataset')
        ground_truth = []
        predict_labels = []
        seq_lengths = []
        for words, labels in batch_iter(dataset, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict(words)
            ground_truth.append(labels)
            predict_labels.append(labels_pred)
            seq_lengths.append(sequence_lengths)
        eval_score = compute_accuracy_f1(ground_truth, predict_labels, seq_lengths, self.config.train_task,
                                         self.config.tag_vocab)
        self.logger.info('accuracy: {:04.2f} -- f1 score: {:04.2f}'.format(eval_score['acc'], eval_score['f1']))
        return eval_score
