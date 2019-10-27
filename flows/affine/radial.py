import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows import BaseTransform

@tf.custom_gradient
def norm(x):
    y = tf.norm(x, axis=-1, keepdims=True)
    def grad(dy):
        return dy * (x / (y + 1e-19))
    return y, grad

class Radial(BaseTransform):
    def __init__(self, **kwargs):
        super(Radial, self).__init__(**kwargs)

    def _r(self, z, z_0):
        return norm(z - z_0) # (B,1)

    def _h(self, alpha, r):
        return 1.0 / (alpha + r) # (B,1)

    def _dh(self, alpha, r):
        return -1.0 / tf.math.square(alpha + r) # (B,1)

    def _beta(self, alpha, beta):
        m = tf.math.log(1.0 + tf.math.exp(beta))
        return -alpha + m

    def forward(self, z, alpha, beta, z_0):
        beta = self._beta(alpha, beta)
        r = self._r(z, z_0)
        h = self._h(alpha, r)
        z_ = z + beta*h*(z - z_0)
        return z_

    def inverse(self, y, alpha, beta, z_0):
        beta = self._beta(alpha, beta)
        yz_norm = norm(y - z_0)
        # solving || y - z_0 || = r + (beta*r)/(alpha + r) in terms of r; hopefully it's correct
        r = 0.5*(-tf.math.sqrt(alpha**2.0 + 2.0*alpha*(beta + yz_norm)+(beta - yz_norm)**2.0) - alpha - beta + yz_norm)
        return (y - z_0) / (r*(1.0 + beta / (alpha + r)))

    def _forward_log_det_jacobian(self, z, alpha, beta, z_0):
        d = tf.shape(z)[1]
        beta = self._beta(alpha, beta)
        r = self._r(z, z_0) # (B,1)
        h = self._h(alpha, r) # (B,1)
        beta_h_p1 = 1.0 + beta*h # (B,1)
        beta_dh_r = beta*self._dh(alpha, r)*r # (B,1)
        return tf.math.pow(beta_h_p1, d - 1.)*(1. + beta_h_p1 + beta_dh_r)

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))
