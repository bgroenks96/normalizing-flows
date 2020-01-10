from .. import GlowFlow, GlowStep
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import itertools as it
import pytest

@pytest.mark.parametrize('layer', [0,1,2])
def test_step_forward_inverse(layer):
    normal_diag = tfp.distributions.MultivariateNormalDiag(loc=np.zeros((1,256,), dtype=np.float32),
                                                           scale_diag=np.ones((1,256,), dtype=np.float32))
    reshape = tfp.bijectors.Reshape((8,8,4))
    glow = GlowStep()
    x = normal_diag.sample()
    x_reshaped = reshape.forward(x)
    # use impl methods for unit under test to avoid caching mechanism
    y = glow._forward(x_reshaped)
    x_ = glow._inverse(y)
    np.testing.assert_array_almost_equal(x_, x_reshaped, decimal=5)
    
@pytest.mark.parametrize(['num_layers', 'depth_per_layer'], list(it.product(range(3), range(4))))
def test_flow_forward_inverse(num_layers, depth_per_layer):
    normal_diag = tfp.distributions.MultivariateNormalDiag(loc=np.zeros((1,256,), dtype=np.float32),
                                                           scale_diag=np.ones((1,256,), dtype=np.float32))
    reshape = tfp.bijectors.Reshape((8,8,4))
    glow = GlowFlow(num_layers=num_layers, depth=depth_per_layer)
    x = normal_diag.sample()
    x_reshaped = reshape.forward(x)
    # use impl methods for unit under test to avoid caching mechanism
    y = glow._forward(x_reshaped)
    x_ = glow._inverse(y)
    np.testing.assert_array_almost_equal(x_, x_reshaped, decimal=5)