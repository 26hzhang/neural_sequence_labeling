import tensorflow as tf
import os


class BaseModel(object):
    def __init__(self, config):
        self.cfg = config
        self.logger = config.logger
        self.sess = None
        self.saver = None

    def reinitialize_weights(self, scope_name):
        """Reinitialize parameters in a scope"""
        variables = tf.contrib.framework.get_variables(scope_name)
        self.sess.run(tf.variables_initializer(variables))

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=self.cfg.max_to_keep)
        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.cfg.ckpt_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        if not os.path.exists(self.cfg.ckpt_path):
            os.makedirs(self.cfg.dir_model)
        self.saver.save(self.sess, self.cfg.ckpt_path + self.cfg.model_name, global_step=epoch)

    def close_session(self):
        self.sess.close()

    @staticmethod
    def variable_summaries(variable):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(variable)
            tf.summary.scalar("mean", mean)  # add mean value
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
            tf.summary.scalar("stddev", stddev)  # add standard deviation value
            tf.summary.scalar("max", tf.reduce_max(variable))  # add maximal value
            tf.summary.scalar("min", tf.reduce_min(variable))  # add minimal value
            tf.summary.histogram("histogram", variable)  # add histogram

    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.cfg.summary_dir, self.sess.graph)

    def _build_train_op(self, lr_method, lr, loss, grad_clip):
        with tf.variable_scope('train_step'):
            if lr_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
            elif lr_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            elif lr_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
            elif lr_method == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
            else:  # default adam optimizer
                if lr_method != 'adam':
                    print('Unknown given optimizing method {}. Using default adam optimizer.'.format(lr_method))
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            if grad_clip is not None:
                grads, vs = zip(*optimizer.compute_gradients(loss))
                # clip by global gradient norm
                grads, _ = tf.clip_by_global_norm(grads, grad_clip)
                # clip by gradient norm, tensorflow will automatically ignore None gradients
                # grads = [None if grad is None else tf.clip_by_norm(grad, grad_clip) for grad in grads]
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)
