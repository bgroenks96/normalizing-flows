import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows import AmortizedTransform

class Planar(AmortizedTransform):
    def __init__(self, input_dims, amortized=True, **kwargs):
        self.d = input_dims
        # parameters u, w, b;
        # u is initialized as the unit vector in R^d
        self.u = tf.Variable(np.ones((self.d, 1)) / np.sqrt(self.d), trainable=not amortized,
                             name=f'u_{self.unique_id}', dtype=tf.float32)
        self.w = tf.Variable(np.random.uniform(0., 1., size=(1, self.d)), trainable=not amortized,
                             name=f'w_{self.unique_id}', dtype=tf.float32)
        self.b = tf.Variable(0.0, name=f'b_{self.unique_id}', dtype=tf.float32, trainable=not amortized)
        self.param_count = np.prod(self.u.shape) + np.prod(self.w.shape) + np.prod(self.b.shape)
        super(Planar, self).__init__(param_count=self.param_count, input_dims=input_dims, **kwargs)
        # define nonlinearity function
        self.h = lambda x: tf.math.tanh(x)
        self.dh = lambda x: 1.0 - tf.square(tf.tanh(x))

    def _amortize(self, args: tf.Tensor):
        assert np.prod(args.shape) == self.param_count
        u, w, b = args[:self.d], args[self.d:-1], args[-1]
        self.u.assign(u)
        self.w.assign(w)
        self.b.assign(b)

    def _alpha(self):
        wu = tf.matmul(self.w, self.u)
        m = -1 + tf.nn.softplus(wu)
        return m - wu

    def _u(self):
        alpha = self._alpha()
        alpha_w = alpha*self.w / tf.reduce_sum(self.w**2.0)
        return self.u + tf.transpose(alpha_w)

    def _forward(self, z):
        wz = tf.matmul(z, self.w, transpose_b=True)
        u = self._u()
        return z + tf.matmul(self.h(wz + self.b), u, transpose_b=True)

    def _inverse(self, y):
        alpha = self._alpha()
        z_para = tf.transpose(alpha*self.w / tf.reduce_sum(self.w**2.0))
        wz_para = tf.matmul(self.w, z_para)
        z_orth = y - z_para - self.h(wz_para + self.b)
        return z_orth + z_para

    def _forward_log_det_jacobian(self, z):
        wz = tf.matmul(z, self.w, transpose_b=True)
        dh_dz = tf.multiply(self.dh(wz + self.b), self.w)
        u = self._u()
        return tf.math.log(tf.math.abs(1.0 + tf.matmul(dh_dz, u)))

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self.inverse(y))
