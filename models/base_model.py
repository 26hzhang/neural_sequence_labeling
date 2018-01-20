import tensorflow as tf
import os


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.sess = None
        self.saver = None

    def reinitialize_weights(self, scope_name):
        """Reinitialize parameters in a scope"""
        variables = tf.contrib.framework.get_variables(scope_name)
        self.sess.run(tf.variables_initializer(variables))

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.config.ckpt_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.ckpt_path + self.config.model_name, global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _build_train_op(self, learning_method, learning_rate, loss, grad_clip):
        with tf.variable_scope('train_step'):
            if learning_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            elif learning_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            elif learning_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            elif learning_method == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            else:  # default adam optimizer
                if learning_method != 'adam':
                    print('Unknown given optimizing method {}. Using default adam optimizer.'.format(learning_method))
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            if grad_clip is not None:
                grads, vs = zip(*optimizer.compute_gradients(loss))
                # clip by global gradient norm
                grads, _ = tf.clip_by_global_norm(grads, grad_clip)
                # clip by gradient norm, tensorflow will automatically ignore None gradients
                # grads = [None if grad is None else tf.clip_by_norm(grad, grad_clip) for grad in grads]
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)
