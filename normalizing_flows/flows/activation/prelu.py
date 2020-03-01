import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from normalizing_flows.flows import Transform

class PReLU(Transform):
    def __init__(self, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.alpha = tf.Variable(0.01, name='alpha', dtype=tf.float32)

    def _forward(self, z):
        alpha = tf.abs(self.alpha)
        ildj = self._inverse_log_det_jacobian(z, alpha)
        return tf.where(z >= 0, z, alpha*z), -ildj

    def _inverse(self, z):
        alpha = tf.abs(self.alpha)
        ildj = self._inverse_log_det_jacobian(z, alpha)
        return tf.where(z >= 0, z, 1.0 / alpha * z), ildj

    def _inverse_log_det_jacobian(self, z, alpha):
        I = tf.ones_like(z)
        inv_jacobian = tf.where(z >= 0, I, I * 1.0 / alpha)
        return tf.reduce_sum(tf.math.log(tf.math.abs(inv_jacobian)))
