import tensorflow as tf
from models.preprocess import get_chunks
import numpy as np


class BiRNN(object):
    def __init__(self, num_units, state_is_tuple=True, cell_type='lstm', scope='bi_rnn'):
        self.num_units = num_units
        if cell_type == 'gru':
            self.cell_fw = tf.nn.rnn_cell.GRUCell(self.num_units)
            self.cell_bw = tf.nn.rnn_cell.GRUCell(self.num_units)
        else:  # default
            self.cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=state_is_tuple)
            self.cell_bw = tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=state_is_tuple)
        self.scope = scope

    def __call__(self, inputs, seq_len, return_last_state=False, keep_prob=None):
        with tf.variable_scope(self.scope):
            if return_last_state:
                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(
                    self.cell_fw, self.cell_bw, inputs, sequence_length=seq_len, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
            else:
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    self.cell_fw, self.cell_bw, inputs, sequence_length=seq_len, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
            if keep_prob is not None:
                output = tf.nn.dropout(output, keep_prob=keep_prob)
        return output


class StackedBiRNN(object):
    def __init__(self, num_layers, num_units, cell_type='lstm', scope='stacked_rnn'):
        self.num_layers = num_layers
        self.num_units = num_units
        if cell_type == 'gru':
            self.cells_fw = [tf.nn.rnn_cell.GRUCell(self.num_units) for _ in range(self.num_layers)]
            self.cells_bw = [tf.nn.rnn_cell.GRUCell(self.num_units) for _ in range(self.num_layers)]
        else:  # default
            self.cells_fw = [tf.nn.rnn_cell.LSTMCell(self.num_units) for _ in range(self.num_layers)]
            self.cells_bw = [tf.nn.rnn_cell.LSTMCell(self.num_units) for _ in range(self.num_layers)]
        self.scope = scope

    def __call__(self, inputs, seq_len, keep_prob=None):
        with tf.variable_scope(self.scope):
            output, *_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                self.cells_fw, self.cells_bw, inputs, sequence_length=seq_len, dtype=tf.float32)
            if keep_prob is not None:
                output = tf.nn.dropout(output, keep_prob)
        return output


def dense(inputs, hidden_dim, use_bias=True, scope='dense'):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden_dim]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        w = tf.get_variable("W", shape=[dim, hidden_dim], dtype=tf.float32)
        output = tf.matmul(flat_inputs, w)
        if use_bias:
            b = tf.get_variable("b", shape=[hidden_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            output = tf.nn.bias_add(output, b)
        output = tf.reshape(output, out_shape)
        return output


def viterbi_decode(logits, trans_params, sequence_lengths, scope=None):
    with tf.variable_scope(scope or 'viterbi_decode'):
        viterbi_sequences = []
        # iterate over the sentences due to no batching in viterbi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]
    return viterbi_sequences


def compute_accuracy_f1(ground_truth, predict_labels, seq_lengths, train_task, tag_vocab, ):
    accs, correct_preds, total_correct, total_preds = [], 0.0, 0.0, 0.0
    for labels, labels_pred, sequence_lengths in zip(ground_truth, predict_labels, seq_lengths):
        for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            accs += [a == b for (a, b) in zip(lab, lab_pred)]
            if train_task != 'pos':
                lab_chunks = set(get_chunks(lab, tag_vocab))
                lab_pred_chunks = set(get_chunks(lab_pred, tag_vocab))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)
    acc = f1 = np.mean(accs)  # if the train task is POS tagging, then do not compute f1 score, only accuracy
    if train_task != 'pos':
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return {"acc": 100 * acc, "f1": 100 * f1}
