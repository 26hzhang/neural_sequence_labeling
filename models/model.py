import tensorflow as tf
from models.base_model import BaseModel
from models import pad_sequences, viterbi_decode, compute_accuracy_f1, batch_iter, Progbar
from models import highway_network, multi_conv1d, BiRNN, StackedBiRNN, dense, dropout
import sys


class SeqLabelModel(BaseModel):
    def __init__(self, config):
        sys.stdout.write('Build model...')
        super(SeqLabelModel, self).__init__(config)
        self._add_placeholders()
        self._build_embeddings_op()
        self._build_model_op()
        self._build_pred_op()
        self._build_loss_op()
        self._build_train_op(self.cfg.lr_method, self.lr, self.loss, self.cfg.grad_clip)
        self.initialize_session()
        sys.stdout.write(' done.\n')

    def _add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        # shape = (batch size)
        self.seq_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")
        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")
        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        # hyper parameters
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self.lr = tf.placeholder(dtype=tf.float32, name="lr")
        self.is_train = tf.placeholder(name='is_train', shape=[], dtype=tf.bool)

    def _get_feed_dict(self, words, is_train, labels=None, lr=None, keep_prob=None):
        # perform padding of the given data
        if self.cfg.use_char_emb:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, pad_tok=0, max_length=None)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, max_length=None, max_length_2=None, nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, pad_tok=0, max_length=None)
            char_ids = word_lengths = None
        # build feed dictionary
        feed_dict = {self.word_ids: word_ids, self.seq_lengths: sequence_lengths, self.is_train: is_train}
        if self.cfg.use_char_emb:
            feed_dict[self.char_ids] = char_ids
            feed_dict[self.word_lengths] = word_lengths
        if labels is not None:
            labels, _ = pad_sequences(labels, pad_tok=0, max_length=None)
            feed_dict[self.labels] = labels
        if lr is not None:
            feed_dict[self.lr] = lr
        if keep_prob is not None:
            feed_dict[self.keep_prob] = keep_prob
        return feed_dict, sequence_lengths

    def _build_embeddings_op(self):
        with tf.variable_scope('words'):
            if self.cfg.use_pretrained:
                _word_embeddings = tf.Variable(self.cfg.glove_embeddings, name='_word_embeddings', dtype=tf.float32,
                                               trainable=self.cfg.finetune_emb)
            else:
                _word_embeddings = tf.get_variable(name='_word_embeddings', dtype=tf.float32, trainable=True,
                                                   shape=[self.cfg.word_vocab_size, self.cfg.word_dim])
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")

        with tf.variable_scope('char_rep_method'):
            if self.cfg.use_char_emb:
                _char_embeddings = tf.get_variable(name='_char_embeddings', dtype=tf.float32, trainable=True,
                                                   shape=[self.cfg.char_vocab_size, self.cfg.char_dim])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="char_embeddings")
                s = tf.shape(char_embeddings)  # [batch size, max length of sentence, max length of word, char_dim]
                if self.cfg.char_rep_method == 'rnn':
                    char_embeddings = tf.reshape(char_embeddings, shape=[s[0] * s[1], s[-2], self.cfg.char_dim])
                    word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])
                    char_bi_rnn = BiRNN(self.cfg.num_units_char, scope='char_rnn')
                    output = char_bi_rnn(char_embeddings, word_lengths, return_last_state=True)
                else:  # cnn model for char representation
                    output = multi_conv1d(char_embeddings, self.cfg.filter_sizes, self.cfg.heights, "VALID",
                                          self.is_train, self.keep_prob, scope="char_cnn")
                # shape = (batch size, max sentence length, char representation size)
                self.char_output = tf.reshape(output, [s[0], s[1], self.cfg.char_out_size])
                word_embeddings = tf.concat([word_embeddings, self.char_output], axis=-1)
        if self.cfg.use_highway:
            with tf.variable_scope("highway"):
                self.word_embeddings = highway_network(word_embeddings, self.cfg.highway_num_layers, bias=True,
                                                       is_train=self.is_train, keep_prob=self.keep_prob)
        else:  # directly dropout before model_op
            self.word_embeddings = dropout(word_embeddings, keep_prob=self.keep_prob, is_train=self.is_train)

    def _build_model_op(self):
        with tf.variable_scope('bidirectional_rnn'):
            if self.cfg.mode_type == 'stack':  # n-layers stacked bidirectional rnn
                rnns = StackedBiRNN(self.cfg.num_layers, self.cfg.num_units, scope='stack_mode')
                output = rnns(self.word_embeddings, self.seq_lengths, keep_prob=self.keep_prob, is_train=self.is_train)

            elif self.cfg.mode_type == 'complex':
                """Experimental Function"""
                fst_rnns = BiRNN(self.cfg.num_units, scope='first_bi_rnn')
                merge_output = fst_rnns(self.word_embeddings, self.seq_lengths)
                # match char output shape to merge_output shape
                char_output = dense(self.char_output, 2 * self.cfg.num_units, scope='project')
                merge_output = tf.add(merge_output, char_output)
                rnns = BiRNN(self.cfg.num_units, scope='complex_mode')  # second layer of bi-directional rnn
                output = rnns(merge_output, self.seq_lengths, keep_prob=self.keep_prob, is_train=self.is_train)

            else:  # default single model
                rnns = BiRNN(self.cfg.num_units, scope='single_mode')
                output = rnns(self.word_embeddings, self.seq_lengths, keep_prob=self.keep_prob, is_train=self.is_train)

        self.logits = dense(output, self.cfg.tag_vocab_size, use_bias=True, scope='project')

    def _build_loss_op(self):
        if self.cfg.use_crf:
            log_ll, self.trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.seq_lengths)
            self.loss = tf.reduce_mean(-log_ll)
        else:  # using softmax
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.seq_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

    def _build_pred_op(self):
        if not self.cfg.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def train(self, train_set, dev_set, test_set):
        self.logger.info('Start training...')
        best_score = 0  # store the current best f1 score on dev_set, updated if new best one is derived
        no_imprv_epoch_count = 0  # count the continuous no improvement epochs
        init_lr = self.cfg.lr  # initial learning rate
        for epoch in range(1, self.cfg.epochs + 1):  # run each epoch
            self.logger.info('Epoch %2d/%2d:' % (epoch, self.cfg.epochs))
            prog = Progbar(target=(len(train_set) + self.cfg.batch_size - 1) // self.cfg.batch_size)  # nbatches
            for i, (words, labels) in enumerate(batch_iter(train_set, self.cfg.batch_size)):
                feed_dict, _ = self._get_feed_dict(words, True, labels, self.cfg.lr, self.cfg.keep_prob)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                prog.update(i + 1, [("train loss", train_loss)])
            self.evaluate(dev_set)  # evaluate dev_set
            metrics = self.evaluate(test_set, eval_dev=False)  # evaluate test_set
            cur_score = metrics['f1']
            # learning rate decay method
            if self.cfg.lr_decay_method == 1:
                self.cfg.lr *= self.cfg.lr_decay
            else:
                self.cfg.lr = init_lr / (1 + self.cfg.lr_decay_rate * epoch)
            if cur_score > best_score:  # performs early stop and parameters save
                no_imprv_epoch_count = 0
                self.save_session(epoch)  # save model with a new best score is obtained
                best_score = cur_score
                self.logger.info(' -- new BEST score: {:04.2f}'.format(best_score))
            else:
                no_imprv_epoch_count += 1
                if no_imprv_epoch_count >= self.cfg.no_imprv_threshold:
                    self.logger.info('early stop at {}th epoch without improvement for {} epochs, BEST score: {:04.2f}'
                                     .format(epoch, no_imprv_epoch_count, best_score))
                    self.save_session(epoch)  # save the last one
                    break
        self.logger.info('Training process done...')

    def predict(self, words):
        feed_dict, sequence_lengths = self._get_feed_dict(words, False, keep_prob=1.0)
        if self.cfg.use_crf:  # get tag scores and transition params of CRF
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
        actuals = []
        predicts = []
        seq_lengths = []
        for words, labels in batch_iter(dataset, self.cfg.batch_size):
            labels_pred, sequence_lengths = self.predict(words)
            actuals.append(labels)
            predicts.append(labels_pred)
            seq_lengths.append(sequence_lengths)
        eval_score = compute_accuracy_f1(actuals, predicts, seq_lengths, self.cfg.train_task, self.cfg.tag_vocab)
        self.logger.info('accuracy: {:04.2f} -- f1 score: {:04.2f}'.format(eval_score['acc'], eval_score['f1']))
        return eval_score
