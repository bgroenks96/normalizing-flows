from flows import Flow
from flows.glow import GlowFlow, GlowStep, Squeeze
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
    np.testing.assert_array_almost_equal(x_, x, decimal=4)
    np.testing.assert_almost_equal(ildj, -fldj, decimal=4)
    
@pytest.mark.parametrize(['num_layers', 'depth_per_layer'], list(it.product(range(1,4), range(1,4))))
def test_flow_forward_inverse(num_layers, depth_per_layer):
    shape = tf.TensorShape((1,8,8,4))
    normal_diag = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(shape, dtype=np.float32),
                                                           scale_diag=np.ones(shape, dtype=np.float32))
    glow = GlowFlow(shape, num_layers=num_layers, depth_per_layer=depth_per_layer)
    x = normal_diag.sample()
    z, fldj = glow.forward(x)
    assert np.all(np.isfinite(z)), 'forward has nan output'
    x_, ildj = glow.inverse(z)
    np.testing.assert_array_almost_equal(x_, x, decimal=4)
    np.testing.assert_almost_equal(ildj, -fldj, decimal=4)
    
def test_flatten_zs():
    shape = tf.TensorShape((10,8,8,1))
    glow = GlowFlow(shape, num_layers=1, depth_per_layer=1)
    squeeze_flow = Flow([Squeeze() for i in range(3)])
    squeeze_flow.initialize(shape)
    z0 = tf.concat([tf.ones(shape), 2*tf.ones(shape), 3*tf.ones(shape)], axis=-1)
    zs, _ = squeeze_flow.forward(z0, return_sequence=True)
    z_flat = glow._flatten_zs(zs[1:])
    np.testing.assert_array_equal(z_flat.shape, (shape[0], np.prod(shape[1:])*9))
    
def test_unflatten_z():
    shape = tf.TensorShape((None,8,8,4))
    n = 3
    normal_diag = tfp.distributions.Normal(loc=np.zeros(shape[1:], dtype=np.float32),
                                           scale=np.ones(shape[1:], dtype=np.float32))
    glow = GlowFlow(shape, num_layers=3, depth_per_layer=1)
    x = normal_diag.sample((n,))
    z_flat, _ = glow.forward(x)
    zs, _ = glow.forward(x, flatten_zs=False)
    z_shapes = [z.shape for z in zs]
    np.testing.assert_equal(3, len(zs))
    np.testing.assert_array_equal(z_flat.shape, (n, np.prod(shape[1:])))
    zs_ = glow._unflatten_z(z_flat)
    z_shapes_ = [z.shape for z in zs_]
    np.testing.assert_array_equal(z_shapes, z_shapes_)
    for z, z_ in zip(zs, zs_):
        np.testing.assert_array_almost_equal(z, z_, decimal=6)