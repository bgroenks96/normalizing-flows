import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.base import Transform

class PReLU(Transform):
    def __init__(self, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.scope = f'prelu_{self.unique_id}'
        self.alpha = tf.Variable(0.01, name='alpha', dtype=tf.float32)

    def _forward(self, z):
        alpha = tf.abs(self.alpha)
        return tf.where(z >= 0, z, alpha*z)

    def _inverse(self, x):
        alpha = tf.abs(self.alpha)
        return tf.where(x >= 0, x, 1.0 / alpha * x)

    def _inverse_log_det_jacobian(self, y):
        I = tf.ones_like(y)
        alpha = tf.abs(self.alpha)
        inv_jacobian = tf.where(y >= 0, I, I * 1.0 / alpha)
        return tf.reduce_sum(tf.math.log(tf.math.abs(inv_jacobian)))
