import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from normalizing_flows.flows import Transform

@tf.custom_gradient
def norm(x):
    y = tf.norm(x, axis=-1, keepdims=True)
    def grad(dy):
        return dy * (x / (y + 1e-19))
    return y, grad

class Radial(Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _r(self, z, z_0):
        return norm(z - z_0) # (B,1)

    def _h(self, alpha, r):
        return 1.0 / (alpha + r) # (B,1)

    def _dh(self, alpha, r):
        return -1.0 / tf.math.square(alpha + r) # (B,1)

    def _beta(self, alpha, beta):
        m = tf.math.log(1.0 + tf.math.exp(beta))
        return -alpha + m

    def _forward(self, z, alpha, beta, z_0):
        d = tf.shape(z)[1]
        beta = self._beta(alpha, beta)
        r = self._r(z, z_0)
        h = self._h(alpha, r)
        z_ = z + beta*h*(z - z_0)
        beta_h_p1 = 1.0 + beta*h # (B,1)
        beta_dh_r = beta*self._dh(alpha, r)*r # (B,1)
        ldj = tf.math.pow(beta_h_p1, d - 1.)*(1. + beta_h_p1 + beta_dh_r)
        return z_, ldj
    
    def _param_count(self, shape):
        d = shape[-1]
        return 2*d + 1