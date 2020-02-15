from .. import InvertibleConv
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def test_forward_inverse():
    shape = tf.TensorShape((1,8,8,2))
    normal_diag = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(shape, dtype=np.float32),
                                                           scale_diag=np.ones(shape, dtype=np.float32))
    conv = InvertibleConv(shape)
    x = normal_diag.sample()
    y, fldj = conv.forward(x)
    x_, ildj = conv.inverse(y)
    np.testing.assert_array_almost_equal(x_, x, decimal=5)
    np.testing.assert_equal(ildj, -fldj)