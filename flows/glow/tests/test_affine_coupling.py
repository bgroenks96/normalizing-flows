from .. import AffineCoupling
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def test_forward_inverse():
    normal_diag = tfp.distributions.MultivariateNormalDiag(loc=np.zeros((1,128,), dtype=np.float32),
                                                           scale_diag=np.ones((1,128,), dtype=np.float32))
    reshape = tfp.bijectors.Reshape((8,8,2))
    affine = AffineCoupling()
    x = normal_diag.sample()
    x_reshaped = reshape.forward(x)
    # use impl methods for unit under test to avoid caching mechanism
    y = affine._forward(x_reshaped)
    x_ = affine._inverse(y)
    np.testing.assert_array_almost_equal(x_, x_reshaped, decimal=5)