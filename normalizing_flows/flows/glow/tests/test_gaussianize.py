from .. import Gaussianize, LogGaussianize
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def test_gaussianize_forward_inverse():
    shape = (1000, 4, 4, 16)
    gaussianize = Gaussianize(input_shape=shape)
    x = tf.random.normal(shape)
    z, fldj = gaussianize.forward(tf.zeros_like(x), x)
    x_, ildj = gaussianize.inverse(tf.zeros_like(x), z)
    np.testing.assert_array_almost_equal(np.mean(z), 0.0, decimal=2)
    np.testing.assert_array_almost_equal(np.std(z), 1.0, decimal=2)
    np.testing.assert_array_almost_equal(x_, x, decimal=5)
    np.testing.assert_array_equal(ildj, -fldj)
    
def test_log_gaussianize_forward_inverse():
    shape = (1000, 4, 4, 16)
    gaussianize = LogGaussianize(input_shape=shape)
    x = tf.math.exp(tf.random.normal(shape))
    z, fldj = gaussianize.forward(tf.zeros_like(x), x)
    x_, ildj = gaussianize.inverse(tf.zeros_like(x), z)
    np.testing.assert_array_almost_equal(np.mean(np.log(z)), 0.0, decimal=2)
    np.testing.assert_array_almost_equal(np.std(np.log(z)), 1.0, decimal=2)
    np.testing.assert_array_almost_equal(x_, x, decimal=4)
    np.testing.assert_array_almost_equal(ildj, -fldj, decimal=4)