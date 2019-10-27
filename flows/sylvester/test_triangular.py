import tensorflow as tf
import numpy as np
from .triangular import TriangularSylvester

BATCH_SIZE = 3

def test_param_count():
    d = 2
    tsnf = TriangularSylvester()
    r = d**2
    diags = 2*d
    bias = d
    assert tsnf.param_count(d) == r + diags + bias

def test_parameterize():
    d = 2
    expected_triu_params = (d**2 + d) // 2
    tsnf = TriangularSylvester()
    param_count = tsnf.param_count(d)
    args = tf.ones((BATCH_SIZE, param_count))
    r1, r2, diag_1, diag_2, b = tsnf._parameterize(tf.constant(d), args)
    # check r1
    assert r1.shape == (BATCH_SIZE, d, d)
    r1_sum = tf.reduce_sum(r1)
    assert r1_sum == BATCH_SIZE*expected_triu_params, f'expected {BATCH_SIZE*expected_triu_params}, got {r1_sum}'
    # check r2
    assert r2.shape == (BATCH_SIZE, d, d)
    r2_sum = tf.reduce_sum(r1)
    assert r2_sum == BATCH_SIZE*expected_triu_params, f'expected {BATCH_SIZE*expected_triu_params}, got {r2_sum}'
    assert diag_1.shape == (BATCH_SIZE, d)
    assert diag_2.shape == (BATCH_SIZE, d)
    assert b.shape == (BATCH_SIZE, 1, d)

def test_diag_r():
    tsnf = TriangularSylvester()
    r = tf.zeros((BATCH_SIZE, 2, 2), dtype=tf.float32)
    diag = tf.ones((BATCH_SIZE, 2), dtype=tf.float32)
    r, diag_out = tsnf._diag_r(r, diag)
    diag_inds = [0,1]
    assert np.all(np.isclose(diag_out, r.numpy()[:,diag_inds,diag_inds]))

def test_permute_identity():
    d = 2
    tsnf = TriangularSylvester()
    I = tf.eye(d, batch_shape=(BATCH_SIZE,))
    z = tf.reshape(tf.range(BATCH_SIZE*d, dtype=tf.float32), (BATCH_SIZE, d))
    z_ = tsnf._permute(z, I)
    assert np.all(z == z_)

def test_permute_reverse():
    d = 2
    tsnf = TriangularSylvester()
    I_r = tf.reverse(tf.eye(d, batch_shape=(BATCH_SIZE,)), axis=(1,))
    z = tf.reshape(tf.range(BATCH_SIZE*d, dtype=tf.float32), (BATCH_SIZE, d))
    z_perm = tsnf._permute(z, I_r)
    z_rec = tsnf._permute(z_perm, I_r)
    assert np.all(tf.reverse(z, axis=(1,)) == z_perm)
    assert np.all(z == z_rec)
