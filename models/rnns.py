import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, RNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from models.nns import flatten, reconstruct, dense


class BiRNN:  # used as encoding or char representation
    def __init__(self, num_units, cell_type='lstm', scope='bi_rnn'):
        self.cell_fw = LSTMCell(num_units) if cell_type == 'lstm' else GRUCell(num_units)
        self.cell_bw = LSTMCell(num_units) if cell_type == 'lstm' else GRUCell(num_units)
        self.scope = scope

    def __call__(self, inputs, seq_len, return_last_state=False, time_major=False):
        assert not time_major, "BiRNN class cannot support time_major currently"
        with tf.variable_scope(self.scope):
            flat_inputs = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [x] (one dimension sequence)
            outputs, ((_, h_fw), (_, h_bw)) = bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, flat_inputs,
                                                                        sequence_length=seq_len, dtype=tf.float32)
            if return_last_state:  # return last states
                output = tf.concat([h_fw, h_bw], axis=-1)  # shape = [-1, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2, remove_shape=1)  # remove the max_time shape
            else:
                output = tf.concat(outputs, axis=-1)  # shape = [-1, max_time, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2)  # reshape to same as inputs, except the last two dim
            return output


class StackBiRNN:
    def __init__(self, num_layers, num_units, cell_type='lstm', scope='stack_bi_rnn'):
        if type(num_units) == list:
            assert len(num_units) == num_layers, "if num_units is a list, then its size should equal to num_layers"
            self.cells_fw = [LSTMCell(num_units[i]) for i in range(num_layers)] if cell_type == 'lstm' else \
                [GRUCell(num_units[i]) for i in range(num_layers)]
            self.cells_bw = [LSTMCell(num_units[i]) for i in range(num_layers)] if cell_type == 'lstm' else \
                [GRUCell(num_units[i]) for i in range(num_layers)]
        else:
            self.cells_fw = [LSTMCell(num_units) for _ in range(num_layers)] if cell_type == 'lstm' else \
                [GRUCell(num_units) for _ in range(num_layers)]
            self.cells_bw = [LSTMCell(num_units) for _ in range(num_layers)] if cell_type == 'lstm' else \
                [GRUCell(num_units) for _ in range(num_layers)]
        self.num_layers = num_layers
        self.scope = scope

    def __call__(self, inputs, seq_len, return_last_state=False, time_major=False):
        assert not time_major, "StackBiRNN class cannot support time_major currently"
        with tf.variable_scope(self.scope):
            flat_inputs = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [x] (one dimension sequence)
            outputs, states_fw, states_bw = stack_bidirectional_dynamic_rnn(self.cells_fw, self.cells_fw, flat_inputs,
                                                                            sequence_length=seq_len, dtype=tf.float32)
            if return_last_state:  # return last states
                # since states_fw is the final states, one tensor per layer, of the forward rnn and states_bw is the
                # final states, one tensor per layer, of the backward rnn, here we extract the last layer of forward
                # and backward states as last state
                h_fw, h_bw = states_fw[self.num_layers - 1].h, states_bw[self.num_layers - 1].h
                output = tf.concat([h_fw, h_bw], axis=-1)  # shape = [-1, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2, remove_shape=1)  # remove the max_time shape
            else:
                output = tf.concat(outputs, axis=-1)  # shape = [-1, max_time, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2)  # reshape to same as inputs, except the last two dim
            return output


class DenseConnectBiRNN:
    """Implement according to Densely Connected Bidirectional LSTM with Applications to Sentence Classification
       https://arxiv.org/pdf/1802.00889.pdf"""
    def __init__(self, num_layers, num_units, cell_type='lstm', scope='dense_connect_bi_rnn'):
        if type(num_units) == list:
            assert len(num_units) == num_layers, "if num_units is a list, then its size should equal to num_layers"
        self.dense_bi_rnn = []
        for i in range(num_layers):
            units = num_units[i] if type(num_units) == list else num_units
            self.dense_bi_rnn.append(BiRNN(units, cell_type, scope='bi_rnn_{}'.format(i)))
        self.num_layers = num_layers
        self.scope = scope

    def __call__(self, inputs, seq_len, time_major=False):
        assert not time_major, "DenseConnectBiRNN class cannot support time_major currently"
        # this function does not support return_last_state method currently
        with tf.variable_scope(self.scope):
            flat_inputs = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [x] (one dimension sequence)
            cur_inputs = flat_inputs
            for i in range(self.num_layers):
                cur_outputs = self.dense_bi_rnn[i](cur_inputs, seq_len)
                if i < self.num_layers - 1:
                    cur_inputs = tf.concat([cur_inputs, cur_outputs], axis=-1)
                else:
                    cur_inputs = cur_outputs
            output = reconstruct(cur_inputs, ref=inputs, keep=2)
            return output


class AttentionCell(RNNCell):  # attention with late fusion
    """Implement of https://pdfs.semanticscholar.org/8785/efdad2abc384d38e76a84fb96d19bbe788c1.pdf?_ga=2.156364859.18139
    40814.1518068648-1853451355.1518068648
    refer: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py"""
    def __init__(self, num_units, memory, pmemory, cell_type='lstm'):
        super(AttentionCell, self).__init__()
        self._cell = LSTMCell(num_units) if cell_type == 'lstm' else GRUCell(num_units)
        self.num_units = num_units
        self.memory = memory
        self.pmemory = pmemory
        self.mem_units = memory.get_shape().as_list()[-1]

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        c, m = state
        # (max_time, batch_size, att_unit)
        ha = tf.nn.tanh(tf.add(self.pmemory, dense(m, self.mem_units, use_bias=False, scope='wah')))
        alphas = tf.squeeze(tf.exp(dense(ha, hidden=1, use_bias=False, scope='way')), axis=[-1])
        alphas = tf.div(alphas, tf.reduce_sum(alphas, axis=0, keep_dims=True))  # (max_time, batch_size)
        # (batch_size, att_units)
        w_context = tf.reduce_sum(tf.multiply(self.memory, tf.expand_dims(alphas, axis=-1)), axis=0)
        h, new_state = self._cell(inputs, state)
        lfc = dense(w_context, self.num_units, use_bias=False, scope='wfc')
        # (batch_size, num_units)
        fw = tf.sigmoid(dense(lfc, self.num_units, use_bias=False, scope='wff') + dense(h, self.num_units, scope='wfh'))
        hft = tf.multiply(lfc, fw) + h  # (batch_size, num_units)
        return hft, new_state
