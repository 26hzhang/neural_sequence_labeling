import tensorflow as tf
from functools import reduce
from operator import mul
from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.python.util import nest


def dense(inputs, hidden, use_bias=True, activation=None, scope="dense"):
    with tf.variable_scope(scope):
        flat_inputs = flatten(inputs, keep=1)
        w = tf.get_variable("weight", [inputs.get_shape().as_list()[-1], hidden], dtype=tf.float32)
        res = tf.matmul(flat_inputs, w)
        if use_bias:
            b = tf.get_variable("bias", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        if activation is not None:
            res = activation(res)
        res = reconstruct(res, ref=inputs, keep=1)
        return res


def highway_layer(arg, bias, bias_start=0.0, scope=None, keep_prob=None, is_train=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', keep_prob=keep_prob, is_train=is_train)
        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', keep_prob=keep_prob, is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, keep_prob=None, is_train=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx),
                                keep_prob=keep_prob, is_train=is_train)
            prev = cur
        return cur


def dot_attention(inputs, memory, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):
        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(dense(d_memory, hidden, use_bias=False, scope="memory"))
            outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (hidden ** 0.5)
            logits = tf.nn.softmax(outputs)
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=-1)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res * gate


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        if keep_prob is not None and is_train is not None:
            out = tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed), lambda: x)
            return out
        return x


def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=None, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype=tf.float32)
        bias = tf.get_variable("bias", shape=[filter_size], dtype=tf.float32)
        strides = [1, 1, 1, 1]
        if is_train is not None and keep_prob is not None:
            in_ = dropout(in_, keep_prob, is_train)
        # [batch, max_len_sent, max_len_word / filter_stride, char output size]
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias
        out = tf.reduce_max(tf.nn.relu(xxc), axis=2)  # max-pooling, [-1, max_len_sent, char output size]
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=None, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for i, (filter_size, height) in enumerate(zip(filter_sizes, heights)):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob,
                         scope="conv1d_{}".format(i))
            outs.append(out)
        concat_out = tf.concat(axis=2, values=outs)
        return concat_out


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, keep_prob=None, is_train=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]
    flat_args = [flatten(arg, 1) for arg in args]
    if keep_prob is not None and is_train is not None:
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, keep_prob), lambda: arg) for arg in flat_args]
    with tf.variable_scope(scope or 'Linear'):
        flat_out = _linear(flat_args, output_size, bias, bias_initializer=tf.constant_initializer(bias_start))
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
    return out


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


def viterbi_decode(logits, trans_params, sequence_lengths, scope=None):
    with tf.variable_scope(scope or 'viterbi_decode'):
        viterbi_sequences = []
        # iterate over the sentences due to no batching in viterbi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]
    return viterbi_sequences
