import tensorflow as tf
import numpy as np
from numpy import nan
import codecs
import os
from models import BaseModel, AttentionCell, highway_network, BiRNN, DenselyConnectedBiRNN, multi_conv1d
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib.crf import crf_log_likelihood
from utils import Progbar, pad_char_sequences
from data.punct_prepro import PUNCTUATION_VOCABULARY, PUNCTUATION_MAPPING, END, UNK, EOS_TOKENS, SPACE


class SequenceLabelModel(BaseModel):
    def __init__(self, config):
        super(SequenceLabelModel, self).__init__(config)

    def _add_placeholders(self):
        self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")  # shape = (batch_size, max_time)
        self.tags = tf.placeholder(tf.int32, shape=[None, None], name="tags")  # shape = (batch_size, max_time - 1)
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        if self.cfg["use_chars"]:
            # shape = (batch_size, max_time, max_word_length)
            self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name="chars")
            self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        # hyper-parameters
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
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

    def _build_embedding_op(self):
        with tf.variable_scope("embeddings"):
            if not self.cfg["use_pretrained"]:
                self.word_embeddings = tf.get_variable(name="emb", dtype=tf.float32, trainable=True,
                                                       shape=[self.word_vocab_size, self.cfg["emb_dim"]])
            else:
                word_emb_1 = tf.Variable(np.load(self.cfg["pretrained_emb"])["embeddings"], name="word_emb_1",
                                         dtype=tf.float32, trainable=self.cfg["tuning_emb"])
                word_emb_2 = tf.get_variable(name="word_emb_2", shape=[3, self.cfg["emb_dim"]], dtype=tf.float32,
                                             trainable=True)  # For UNK, NUM and END
                self.word_embeddings = tf.concat([word_emb_1, word_emb_2], axis=0)
            word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.words, name="word_emb")
            print("word embedding shape: {}".format(word_emb.get_shape().as_list()))
            if self.cfg["use_chars"]:
                self.char_embeddings = tf.get_variable(name="c_emb", dtype=tf.float32, trainable=True,
                                                       shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
                char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.chars, name="chars_emb")
                # train char representation
                if self.cfg["char_represent_method"] == "rnn":
                    char_bi_rnn = BiRNN(self.cfg["char_num_units"], cell_type=self.cfg["cell_type"], scope="c_bi_rnn")
                    char_represent = char_bi_rnn(char_emb, self.char_seq_len, use_last_state=True)
                else:
                    char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
                                                  drop_rate=self.drop_rate,
                                                  is_train=self.is_train)
                print("chars representation shape: {}".format(char_represent.get_shape().as_list()))
                word_emb = tf.concat([word_emb, char_represent], axis=-1)
            if self.cfg["use_highway"]:
                self.word_emb = highway_network(word_emb, self.cfg["highway_layers"], use_bias=True, bias_init=0.0,
                                                keep_prob=self.keep_prob, is_train=self.is_train)
            else:
                self.word_emb = tf.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
            print("word and chars concatenation shape: {}".format(self.word_emb.get_shape().as_list()))

    def _build_model_op(self):
        with tf.variable_scope("densely_connected_bi_rnn"):
            dense_bi_rnn = DenselyConnectedBiRNN(self.cfg["num_layers"], self.cfg["num_units_list"],
                                                 cell_type=self.cfg["cell_type"])
            context = dense_bi_rnn(self.word_emb, seq_len=self.seq_len)
            print("densely connected bi_rnn output shape: {}".format(context.get_shape().as_list()))

        with tf.variable_scope("attention"):
            p_context = tf.layers.dense(context, units=2 * self.cfg["num_units_list"][-1], use_bias=True,
                                        bias_initializer=tf.constant_initializer(0.0))
            context = tf.transpose(context, [1, 0, 2])
            p_context = tf.transpose(p_context, [1, 0, 2])
            attn_cell = AttentionCell(self.cfg["num_units_list"][-1], context, p_context)
            attn_outs, _ = dynamic_rnn(attn_cell, context[1:, :, :], sequence_length=self.seq_len - 1, dtype=tf.float32,
                                       time_major=True)
            attn_outs = tf.transpose(attn_outs, [1, 0, 2])
            print("attention output shape: {}".format(attn_outs.get_shape().as_list()))

        with tf.variable_scope("project"):
            self.logits = tf.layers.dense(attn_outs, units=self.tag_vocab_size, use_bias=True,
                                          bias_initializer=tf.constant_initializer(0.0))
            print("logits shape: {}".format(self.logits.get_shape().as_list()))

    def _build_loss_op(self):
        if self.cfg["use_crf"]:
            crf_loss, self.trans_params = crf_log_likelihood(self.logits, self.tags, self.seq_len - 1)
            self.loss = tf.reduce_mean(-crf_loss)
        else:  # using softmax
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tags)
            mask = tf.sequence_mask(self.seq_len)
            self.loss = tf.reduce_mean(tf.boolean_mask(losses, mask))
        if self.cfg["l2_reg"] is not None and self.cfg["l2_reg"] > 0.0:  # l2 regularization
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name])
            self.loss += self.cfg["l2_reg"] * l2_loss
        tf.summary.scalar("loss", self.loss)

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        if self.cfg["use_crf"]:
            logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=feed_dict)
            return self.viterbi_decode(logits, trans_params, data["seq_len"] - 1)
        else:
            pred_logits = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
            logits = self.sess.run(pred_logits, feed_dict=feed_dict)
            return logits

    def train_epoch(self, train_set, valid_data, epoch):
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        total_cost, total_samples = 0, 0
        for i, batch in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch, is_train=True, keep_prob=self.cfg["keep_prob"], lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            total_cost += train_loss
            total_samples += np.array(batch["words"]).shape[0]
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss),
                                ("Perplexity", np.exp(total_cost / total_samples))])
            self.train_writer.add_summary(summary, cur_step)
            if i % 100 == 0:
                valid_feed_dict = self._get_feed_dict(valid_data)
                valid_summary = self.sess.run(self.summary, feed_dict=valid_feed_dict)
                self.test_writer.add_summary(valid_summary, cur_step)

    def train(self, train_set, valid_data, valid_text, test_texts):  # test_texts: [ref, asr]
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch = -np.inf, 0
        self._add_summary()
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, self.cfg["epochs"]))
            self.train_epoch(train_set, valid_data, epoch)
            # self.evaluate(valid_text)
            ref_f1 = self.evaluate_punct(test_texts[0])["F1"] * 100.0  # use ref to compute best F1
            asr_f1 = self.evaluate_punct(test_texts[1])["F1"] * 100.0
            if ref_f1 >= best_f1:
                best_f1 = ref_f1
                no_imprv_epoch = 0
                self.save_session(epoch)
                self.logger.info(" -- new BEST score on ref dataset: {:04.2f}, on asr dataset: {:04.2f}"
                                 .format(best_f1, asr_f1))
            else:
                no_imprv_epoch += 1
                if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                    self.logger.info("early stop at {}th epoch without improvement, BEST score on ref dataset: {:04.2f}"
                                     .format(epoch, best_f1))
                    break
        self.train_writer.close()
        self.test_writer.close()

    def evaluate_punct(self, file):
        save_path = os.path.join(self.cfg["checkpoint_path"], "result.txt")
        with codecs.open(file, mode="r", encoding="utf-8") as f:
            text = f.read().split()
        text = [w for w in text if w not in self.tag_dict and w not in PUNCTUATION_MAPPING] + [END]
        index = 0
        with codecs.open(save_path, mode="w", encoding="utf-8") as f_out:
            while True:
                subseq = text[index: index + self.cfg["max_sequence_len"]]
                if len(subseq) == 0:
                    break
                # create feed data
                cvrt_seq = np.array([[self.word_dict.get(w, self.word_dict[UNK]) for w in subseq]], dtype=np.int32)
                seq_len = np.array([len(v) for v in cvrt_seq], dtype=np.int32)
                cvrt_seq_chars = []
                for word in subseq:
                    chars = [self.char_dict.get(c, self.char_dict[UNK]) for c in word]
                    cvrt_seq_chars.append(chars)
                cvrt_seq_chars, char_seq_len = pad_char_sequences([cvrt_seq_chars])
                cvrt_seq_chars = np.array(cvrt_seq_chars, dtype=np.int32)
                char_seq_len = np.array(char_seq_len, dtype=np.int32)
                data = {"words": cvrt_seq, "seq_len": seq_len, "chars": cvrt_seq_chars, "char_seq_len": char_seq_len,
                        "batch_size": 1}
                # predict
                predicts = self._predict_op(data)
                # write to file
                f_out.write(subseq[0])
                last_eos_idx = 0
                punctuations = []
                for preds_t in predicts[0]:
                    punctuation = self.rev_tag_dict[preds_t]
                    punctuations.append(punctuation)
                    if punctuation in EOS_TOKENS:
                        last_eos_idx = len(punctuations)
                if subseq[-1] == END:
                    step = len(subseq) - 1
                elif last_eos_idx != 0:
                    step = last_eos_idx
                else:
                    step = len(subseq) - 1
                for j in range(step):
                    f_out.write(" " + punctuations[j] + " " if punctuations[j] != SPACE else " ")
                    if j < step - 1:
                        f_out.write(subseq[1 + j])
                if subseq[-1] == END:
                    break
                index += step
        out_str, f1, err, ser = self.compute_score(file, save_path)
        score = {"F1": f1, "ERR": err, "SER": ser}
        self.logger.info("\nEvaluate on {}:\n{}\n".format(file, out_str))
        try:  # delete output file after compute scores
            os.remove(save_path)
        except OSError:
            pass
        return score

    def inference(self, sentence):
        pass  # TODO

    @staticmethod
    def compute_score(target_path, predicted_path):
        """Computes and prints the overall classification error and precision, recall, F-score over punctuations."""
        mappings, counter, t_i, p_i = {}, 0, 0, 0
        total_correct, correct, substitutions, deletions, insertions = 0, 0.0, 0.0, 0.0, 0.0
        true_pos, false_pos, false_neg = {}, {}, {}
        with codecs.open(target_path, "r", "utf-8") as f_target, codecs.open(predicted_path, "r", "utf-8") as f_predict:
            target_stream = f_target.read().split()
            predict_stream = f_predict.read().split()
            while True:
                if PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in PUNCTUATION_VOCABULARY:
                    # skip multiple consecutive punctuations
                    target_punct = " "
                    while PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in PUNCTUATION_VOCABULARY:
                        target_punct = PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i])
                        target_punct = mappings.get(target_punct, target_punct)
                        t_i += 1
                else:
                    target_punct = " "
                if predict_stream[p_i] in PUNCTUATION_VOCABULARY:
                    predicted_punct = mappings.get(predict_stream[p_i], predict_stream[p_i])
                    p_i += 1
                else:
                    predicted_punct = " "
                is_correct = target_punct == predicted_punct
                counter += 1
                total_correct += is_correct
                if predicted_punct == " " and target_punct != " ":
                    deletions += 1
                elif predicted_punct != " " and target_punct == " ":
                    insertions += 1
                elif predicted_punct != " " and target_punct != " " and predicted_punct == target_punct:
                    correct += 1
                elif predicted_punct != " " and target_punct != " " and predicted_punct != target_punct:
                    substitutions += 1
                true_pos[target_punct] = true_pos.get(target_punct, 0.0) + float(is_correct)
                false_pos[predicted_punct] = false_pos.get(predicted_punct, 0.) + float(not is_correct)
                false_neg[target_punct] = false_neg.get(target_punct, 0.) + float(not is_correct)
                assert target_stream[t_i] == predict_stream[p_i] or predict_stream[p_i] == "<unk>", \
                    "File: %s \nError: %s (%s) != %s (%s) \nTarget context: %s \nPredicted context: %s" % \
                    (target_path, target_stream[t_i], t_i, predict_stream[p_i], p_i,
                     " ".join(target_stream[t_i - 2:t_i + 2]), " ".join(predict_stream[p_i - 2:p_i + 2]))
                t_i += 1
                p_i += 1
                if t_i >= len(target_stream) - 1 and p_i >= len(predict_stream) - 1:
                    break
        overall_tp, overall_fp, overall_fn = 0.0, 0.0, 0.0
        out_str = "-" * 46 + "\n"
        out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")
        for p in PUNCTUATION_VOCABULARY:
            if p == SPACE:
                continue
            overall_tp += true_pos.get(p, 0.0)
            overall_fp += false_pos.get(p, 0.0)
            overall_fn += false_neg.get(p, 0.0)
            punctuation = p
            precision = (true_pos.get(p, 0.0) / (true_pos.get(p, 0.0) + false_pos[p])) if p in false_pos else nan
            recall = (true_pos.get(p, 0.0) / (true_pos.get(p, 0.0) + false_neg[p])) if p in false_neg else nan
            f_score = (2. * precision * recall / (precision + recall)) if (precision + recall) > 0 else nan
            out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(punctuation, "{:.2f}".format(precision * 100),
                                                            "{:.2f}".format(recall * 100),
                                                            "{:.2f}".format(f_score * 100))
        out_str += "-" * 46 + "\n"
        pre = overall_tp / (overall_tp + overall_fp) if overall_fp else nan
        rec = overall_tp / (overall_tp + overall_fn) if overall_fn else nan
        f1 = (2. * pre * rec) / (pre + rec) if (pre + rec) else nan
        out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("Overall", "{:.2f}".format(pre * 100),
                                                       "{:.2f}".format(rec * 100), "{:.2f}".format(f1 * 100))
        err = round((100.0 - float(total_correct) / float(counter - 1) * 100.0), 2)
        ser = round((substitutions + deletions + insertions) / (correct + substitutions + deletions) * 100, 1)
        out_str += "ERR: %s%%\n" % err
        out_str += "SER: %s%%" % ser
        return out_str, f1, err, ser
