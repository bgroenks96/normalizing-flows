import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.base import BaseTransform

class Radial(BaseTransform):
    def __init__(self, input_dims, **kwargs):
        super(Radial, self).__init__(input_dims=input_dims, **kwargs)
        self.d = input_dims
        self.z_0 = tf.Variable(np.random.uniform(0.1, 1., size=(1, self.d)), name=f'z_0_{self.unique_id}', dtype=tf.float32)
        self.alpha = tf.Variable(0.01, name=f'alpha_{self.unique_id}', dtype=tf.float32,
                                constraint=lambda x: tf.clip_by_value(x, 0.0, np.infty))
        self.beta = tf.Variable(1.0, name=f'beta_{self.unique_id}', dtype=tf.float32, trainable=False)
        self.r = lambda z: tf.norm(z - self.z_0, axis=-1, keepdims=True) # (B,1)
        self.h = lambda alpha, r: 1.0 / (alpha + r) # (B,1)
        self.dh = lambda alpha, r: -1.0 / tf.math.square(alpha + r) # (B,1)


    def _backward(self):
        m = tf.math.log(1.0 + tf.math.exp(self.beta))
        self.beta.assign(-self.alpha + m)
        super(Radial, self)._backward()

    def _forward(self, z):
        assert z.shape[-1] == self.d
        alpha = self.alpha
        r = self.r(z)
        h = self.h(alpha, r)
        y = z + self.beta*h*(z - self.z_0)
        assert y.shape == z.shape
        return y


    def _inverse(self, y):
        alpha = self.alpha
        beta = self.beta
        yz_norm = tf.norm(y - self.z_0)
        # solving || y - z_0 || = r + (beta*r)/(alpha + r) in terms of r; hopefully it's correct
        r = 0.5*(-tf.math.sqrt(alpha**2.0 + 2.0*alpha*(beta + yz_norm)+(beta - yz_norm)**2.0) - alpha - beta + yz_norm)
        with tf.control_dependencies([
            tf.debugging.assert_all_finite(alpha, 'alpha nan'),
            tf.debugging.assert_all_finite(beta, 'beta nan'),
            tf.debugging.assert_all_finite(self.z_0, 'z_0 nan'),
            tf.debugging.assert_all_finite(yz_norm, 'yz_norm nan'),
            tf.debugging.assert_all_finite(r, 'r nan')]):
            return (y - self.z_0) / (r*(1.0 + beta / (alpha + r)))

    def _forward_log_det_jacobian(self, z):
        assert z.shape[-1] == self.d
        alpha = self.alpha
        r = self.r(z) # (B,1)
        h = self.h(alpha, r) # (B,1)
        beta_h_p1 = 1.0 + self.beta*h # (B,1)
        beta_dh_r = self.beta*self.dh(alpha, r)*r # (B,1)
        return tf.math.pow(beta_h_p1, self.d - 1.)*(1. + beta_h_p1 + beta_dh_r)

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))
