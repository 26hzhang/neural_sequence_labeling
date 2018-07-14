import tensorflow as tf
import math
from functools import reduce
from operator import mul
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMCell, GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


def layer_normalize(inputs, epsilon=1e-8, scope=None):
    with tf.variable_scope(scope or "layer_norm"):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = tf.add(tf.multiply(gamma, normalized), beta)
        return outputs


def highway_layer(inputs, use_bias=True, bias_init=0.0, keep_prob=1.0, is_train=False, scope=None):
    with tf.variable_scope(scope or "highway_layer"):
        hidden = inputs.get_shape().as_list()[-1]
        with tf.variable_scope("trans"):
            trans = tf.layers.dropout(inputs, rate=1.0 - keep_prob, training=is_train)
            trans = tf.layers.dense(trans, units=hidden, use_bias=use_bias, bias_initializer=tf.constant_initializer(
                bias_init), activation=None)
            trans = tf.nn.relu(trans)
        with tf.variable_scope("gate"):
            gate = tf.layers.dropout(inputs, rate=1.0 - keep_prob, training=is_train)
            gate = tf.layers.dense(gate, units=hidden, use_bias=use_bias, bias_initializer=tf.constant_initializer(
                bias_init), activation=None)
            gate = tf.nn.sigmoid(gate)
        outputs = gate * trans + (1 - gate) * inputs
        return outputs


def highway_network(inputs, highway_layers=2, use_bias=True, bias_init=0.0, keep_prob=1.0, is_train=False, scope=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = inputs
        cur = None
        for idx in range(highway_layers):
            cur = highway_layer(prev, use_bias, bias_init, keep_prob, is_train, scope="highway_layer_{}".format(idx))
            prev = cur
        return cur


def conv1d(in_, filter_size, height, padding, is_train=True, drop_rate=0.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype=tf.float32)
        bias = tf.get_variable("bias", shape=[filter_size], dtype=tf.float32)
        strides = [1, 1, 1, 1]
        in_ = tf.layers.dropout(in_, rate=drop_rate, training=is_train)
        # [batch, max_len_sent, max_len_word / filter_stride, char output size]
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias
        out = tf.reduce_max(tf.nn.relu(xxc), axis=2)  # max-pooling, [-1, max_len_sent, char output size]
        return out


def multi_conv1d(in_, filter_sizes, heights, padding="VALID", is_train=True, drop_rate=0.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for i, (filter_size, height) in enumerate(zip(filter_sizes, heights)):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, drop_rate=drop_rate,
                         scope="conv1d_{}".format(i))
            outs.append(out)
        concat_out = tf.concat(axis=2, values=outs)
        return concat_out


class BiRNN:
    def __init__(self, num_units, cell_type='lstm', scope=None):
        self.cell_fw = GRUCell(num_units) if cell_type == 'gru' else LSTMCell(num_units)
        self.cell_bw = GRUCell(num_units) if cell_type == 'gru' else LSTMCell(num_units)
        self.scope = scope or "bi_rnn"

    def __call__(self, inputs, seq_len, use_last_state=False, time_major=False):
        assert not time_major, "BiRNN class cannot support time_major currently"
        with tf.variable_scope(self.scope):
            flat_inputs = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [x] (one dimension sequence)
            outputs, ((_, h_fw), (_, h_bw)) = bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, flat_inputs,
                                                                        sequence_length=seq_len, dtype=tf.float32)
            if use_last_state:  # return last states
                output = tf.concat([h_fw, h_bw], axis=-1)  # shape = [-1, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2, remove_shape=1)  # remove the max_time shape
            else:
                output = tf.concat(outputs, axis=-1)  # shape = [-1, max_time, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2)  # reshape to same as inputs, except the last two dim
            return output


class DenselyConnectedBiRNN:
    """Implement according to Densely Connected Bidirectional LSTM with Applications to Sentence Classification
       https://arxiv.org/pdf/1802.00889.pdf"""
    def __init__(self, num_layers, num_units, cell_type='lstm', scope=None):
        if type(num_units) == list:
            assert len(num_units) == num_layers, "if num_units is a list, then its size should equal to num_layers"
        self.dense_bi_rnn = []
        for i in range(num_layers):
            units = num_units[i] if type(num_units) == list else num_units
            self.dense_bi_rnn.append(BiRNN(units, cell_type, scope='bi_rnn_{}'.format(i)))
        self.num_layers = num_layers
        self.scope = scope or "densely_connected_bi_rnn"

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


def multi_head_attention(queries, keys, num_heads, attention_size, drop_rate=0.0, is_train=True, reuse=None,
                         scope=None):
    # borrowed from: https://github.com/Kyubyong/transformer/blob/master/modules.py
    with tf.variable_scope(scope or "multi_head_attention", reuse=reuse):
        if attention_size is None:
            attention_size = queries.get_shape().as_list()[-1]
        # linear projections, shape=(batch_size, max_time, attention_size)
        query = tf.layers.dense(queries, attention_size, activation=tf.nn.relu, name="query_project")
        key = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="key_project")
        value = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="value_project")
        # split and concatenation, shape=(batch_size * num_heads, max_time, attention_size / num_heads)
        query_ = tf.concat(tf.split(query, num_heads, axis=2), axis=0)
        key_ = tf.concat(tf.split(key, num_heads, axis=2), axis=0)
        value_ = tf.concat(tf.split(value, num_heads, axis=2), axis=0)
        # multiplication
        attn_outs = tf.matmul(query_, tf.transpose(key_, [0, 2, 1]))
        # scale
        attn_outs = attn_outs / (key_.get_shape().as_list()[-1] ** 0.5)
        # key masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # shape=(batch_size, max_time)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # shape=(batch_size * num_heads, max_time)
        # shape=(batch_size * num_heads, max_time, max_time)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(attn_outs) * (-2 ** 32 + 1)
        # shape=(batch_size, max_time, attention_size)
        attn_outs = tf.where(tf.equal(key_masks, 0), paddings, attn_outs)
        # activation
        attn_outs = tf.nn.softmax(attn_outs)
        # query masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        attn_outs *= query_masks
        # dropout
        attn_outs = tf.layers.dropout(attn_outs, rate=drop_rate, training=is_train)
        # weighted sum
        outputs = tf.matmul(attn_outs, value_)
        # restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries  # residual connection
        outputs = layer_normalize(outputs)
    return outputs


def dot_attention(inputs, memory, hidden, drop_rate=0.0, is_train=True, scope=None):
    with tf.variable_scope(scope or "dot_attention"):
        d_inputs = tf.layers.dropout(inputs, rate=drop_rate, training=tf.convert_to_tensor(is_train))
        d_memory = tf.layers.dropout(memory, rate=drop_rate, training=tf.convert_to_tensor(is_train))

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(tf.layers.dense(d_inputs, hidden, use_bias=False, name='inputs'))
            memory_ = tf.nn.relu(tf.layers.dense(d_memory, hidden, use_bias=False, name='memory'))
            outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (hidden ** 0.5)
            logits = tf.nn.softmax(outputs)
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=-1)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = tf.layers.dropout(res, rate=drop_rate, training=tf.convert_to_tensor(is_train))
            gate = tf.nn.sigmoid(tf.layers.dense(d_res, dim, use_bias=False, name='gate'))
            return res * gate


class AttentionCell(RNNCell):  # time_major based
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

    def compute_output_shape(self, input_shape):
        pass

    def __call__(self, inputs, state, scope=None):
        c, m = state
        # (max_time, batch_size, att_unit)
        ha = tf.nn.tanh(tf.add(self.pmemory, tf.layers.dense(m, self.mem_units, use_bias=False, name="wah")))
        alphas = tf.squeeze(tf.exp(tf.layers.dense(ha, units=1, use_bias=False, name='way')), axis=[-1])
        alphas = tf.div(alphas, tf.reduce_sum(alphas, axis=0, keepdims=True))  # (max_time, batch_size)
        # (batch_size, att_units)
        w_context = tf.reduce_sum(tf.multiply(self.memory, tf.expand_dims(alphas, axis=-1)), axis=0)
        h, new_state = self._cell(inputs, state)
        lfc = tf.layers.dense(w_context, self.num_units, use_bias=False, name='wfc')
        # (batch_size, num_units)
        fw = tf.sigmoid(tf.layers.dense(lfc, self.num_units, use_bias=False, name='wff') +
                        tf.layers.dense(h, self.num_units, name='wfh'))
        hft = tf.multiply(lfc, fw) + h  # (batch_size, num_units)
        return hft, new_state


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different frequency and phase.
    This allows attention to learn to use absolute and relative positions. Timing signals should be added to some
    precursors of both the query and the memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be expressed in terms of y,
    sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with min_timescale and ending with max_timescale.
    The number of different timescales is equal to channels / 2. For each timescale, we generate the two sinusoidal
    signals sin(timestep/timescale) and cos(timestep/timescale).  All of these sinusoids are concatenated in the
    channels dimension.
    Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float
    Returns:
        a Tensor the same shape as x.
    """
    with tf.name_scope("add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) /
                                   (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)

        scaled_time = (tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal


def label_smoothing(inputs, epsilon=0.1):
    dim = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / dim)


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, remove_shape=None):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    if remove_shape is not None:
        tensor_start = tensor_start + remove_shape
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out
