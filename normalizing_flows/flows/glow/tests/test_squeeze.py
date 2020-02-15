from .. import Squeeze
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def test_forward_inverse():
    shape = tf.TensorShape((1,8,8,2))
    normal_diag = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(shape, dtype=np.float32),
                                                           scale_diag=np.ones(shape, dtype=np.float32))
    squeeze = Squeeze(shape, factor=2)
    x = normal_diag.sample()
    y,_ = squeeze.forward(x)
    np.testing.assert_array_equal(y.shape, [1,4,4,8])
    x_,_ = squeeze.inverse(y)
    np.testing.assert_array_equal(x_.shape, x.shape)
    
def test_forward_inverse_indivisible_shape():
    shape = tf.TensorShape((1,9,9,2))
    normal_diag = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(shape, dtype=np.float32),
                                                           scale_diag=np.ones(shape, dtype=np.float32))
    squeeze = Squeeze(shape, factor=2)
    x = normal_diag.sample()
    y,_ = squeeze.forward(x)
    np.testing.assert_array_equal(y.shape, [1,5,5,8])
    x_,_ = squeeze.inverse(y)
    np.testing.assert_array_equal(x_.shape, x.shape)