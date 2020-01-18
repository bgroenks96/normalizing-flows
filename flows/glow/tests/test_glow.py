from .. import GlowFlow, GlowStep
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import itertools as it
import pytest

@pytest.mark.parametrize('layer', [1,2,3])
def test_step_forward_inverse(layer):
    shape = tf.TensorShape((1,8,8,4))
    normal_diag = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(shape, dtype=np.float32),
                                                           scale_diag=np.ones(shape, dtype=np.float32))
    glow = GlowStep(shape)
    x = normal_diag.sample()
    y, fldj = glow.forward(x)
    x_, ildj = glow.inverse(y)
    np.testing.assert_array_almost_equal(x_, x, decimal=5)
    np.testing.assert_almost_equal(ildj, -fldj, decimal=5)
    
@pytest.mark.parametrize(['num_layers', 'depth_per_layer'], list(it.product(range(1,4), range(1,4))))
def test_flow_forward_inverse(num_layers, depth_per_layer):
    shape = tf.TensorShape((1,8,8,4))
    normal_diag = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(shape, dtype=np.float32),
                                                           scale_diag=np.ones(shape, dtype=np.float32))
    glow = GlowFlow(shape, num_layers=num_layers, depth=depth_per_layer)
    x = normal_diag.sample()
    y, fldj = glow.forward(x)
    assert np.all(np.isfinite(y)), 'forward has nan output'
    x_, ildj = glow.inverse(y)
    np.testing.assert_array_almost_equal(x_, x, decimal=5)
    np.testing.assert_almost_equal(ildj, -fldj, decimal=5)