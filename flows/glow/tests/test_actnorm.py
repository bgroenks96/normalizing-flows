from .. import ActNorm
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def test_forward_inverse():
    normal_diag = tfp.distributions.MultivariateNormalDiag(loc=np.zeros((1,64,), dtype=np.float32),
                                                           scale_diag=np.ones((1,64,), dtype=np.float32))
    actnorm = ActNorm(event_ndims=1)
    x = normal_diag.sample()
    # use impl methods for unit under test to avoid caching mechanism
    y = actnorm._forward(x)
    x_ = actnorm._inverse(y)
    np.testing.assert_array_almost_equal(x_, x, decimal=5)