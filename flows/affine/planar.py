import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.base import BaseTransform

class Planar(BaseTransform):
    def __init__(self, input_dims, **kwargs):
        super(Planar, self).__init__(input_dims=input_dims, **kwargs)
        self.d = input_dims
        # parameters u, w, b;
        # u is initialized as the unit vector in R^d
        self.u = tf.Variable(np.ones((self.d, 1)) / np.sqrt(self.d), name=f'u_{self.unique_id}', dtype=tf.float32)
        self.w = tf.Variable(np.random.uniform(0., 1., size=(self.d, 1)), name=f'w_{self.unique_id}', dtype=tf.float32)
        self.b = tf.Variable(0.0, name=f'b_{self.unique_id}', dtype=tf.float32)
        # define nonlinearity function
        self.h = lambda x: tf.math.tanh(x)
        self.dh = lambda x: 1.0 - tf.square(tf.tanh(x))

    def _alpha(self):
        wu = tf.matmul(self.w, self.u, transpose_a=True)
        m = -1 + tf.math.log(1.0 + tf.math.exp(wu))
        return m - wu
    
    def _u(self):
        alpha = self._alpha()
        alpha_w = alpha*self.w / tf.reduce_sum(self.w**2.0)
        return self.u + alpha_w

    def _forward(self, z):
        wz = tf.matmul(z, self.w)
        u = self._u()
        return z + tf.matmul(self.h(wz + self.b), u, transpose_b=True)

    def _inverse(self, y):
        alpha = self._alpha()
        z_para = alpha*self.w / tf.reduce_sum(self.w**2.0)
        wz_para = tf.matmul(self.w, z_para, transpose_a=True)
        z_orth = y - tf.transpose(z_para) - self.h(wz_para + self.b)
        return z_orth + tf.transpose(z_para)

    def _forward_log_det_jacobian(self, z):
        wz = tf.matmul(z, self.w)
        dh_dz = tf.matmul(self.dh(wz + self.b), self.w, transpose_b=True)
        u = self._u()
        return tf.math.abs(1.0 + tf.matmul(dh_dz, u))

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))
