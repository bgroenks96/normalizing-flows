import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
from .utils import tf_init_var

class InvertibleConv(tfp.bijectors.Bijector):
    def __init__(self, name='invertible_1x1_conv', *args, **kwargs):
        super().__init__(*args, forward_min_event_ndims=3, inverse_min_event_ndims=3, name=name, **kwargs)
        self.P = tf_init_var(event_ndims=2, unspecified_axes=[1,2], trainable=False)
        self.L = tf_init_var(event_ndims=2, unspecified_axes=[1,2])
        self.U = tf_init_var(event_ndims=2, unspecified_axes=[1,2])
        self.log_d = tf_init_var(event_ndims=1, unspecified_axes=[1])
        self.sgn_d = tf_init_var(event_ndims=1, unspecified_axes=[1], trainable=False)
        self.init = False
        
    def _init_vars(self, x):
        if not self.init:
            input_shape = tf.shape(x)
            assert len(input_shape) == 4, 'input should be 4-dimensional'
            batch_size, wt, ht, c = input_shape
            # sample random orthogonal matrix and compute LU decomposition
            q,_ = np.linalg.qr(np.random.randn(c, c))
            p, l, u = scipy.linalg.lu(q)
            d = np.diag(u)
            # parameterize diagonal d as log(d) for numerical stability
            log_d = np.log(np.abs(d))
            sgn_d = np.sign(d)
            l, u = np.tril(l, k=-1), np.triu(u, k=1)
            # initialize variables
            self.input_shape = input_shape
            self.P.assign(np.expand_dims(p, axis=0).astype(np.float32))
            self.L.assign(np.expand_dims(l, axis=0).astype(np.float32))
            self.U.assign(np.expand_dims(u, axis=0).astype(np.float32))
            self.log_d.assign(np.expand_dims(log_d, axis=0).astype(np.float32))
            self.sgn_d.assign(np.expand_dims(sgn_d, axis=0).astype(np.float32))
            self.tril_mask = tf.constant(np.tril(np.ones((1,c,c)), k=-1), dtype=tf.float32)
            self.triu_mask = tf.constant(np.triu(np.ones((1,c,c)), k=1), dtype=tf.float32)
            self.init = True
    
    @tf.function
    def _compute_w(self, l, u, p, log_d, sgn_d):
        d = tf.linalg.diag(tf.math.exp(log_d)*sgn_d)
        l = self.tril_mask*l + tf.eye(self.input_shape[-1])
        u = self.triu_mask*u + d
        w = tf.linalg.matmul(p, tf.linalg.matmul(l, u))
        return tf.expand_dims(w, axis=0) # (1,1,c,c)
    
    @tf.function
    def _compute_w_inverse(self, l, u, p, log_d, sgn_d):
        d = tf.linalg.diag(tf.math.exp(log_d)*sgn_d)
        l_inv = tf.linalg.inv(self.tril_mask*l + tf.eye(self.input_shape[-1]))
        u_inv = tf.linalg.inv(self.triu_mask*u + d)
        p_inv = tf.linalg.inv(p)
        w_inv = tf.linalg.matmul(u_inv, tf.linalg.matmul(l_inv, p_inv))
        return tf.expand_dims(w_inv, axis=0) # (1,1,c,c)
    
    def _forward(self, x):
        self._init_vars(x)
        w = self._compute_w(self.L, self.U, self.P, self.log_d, self.sgn_d)
        y = tf.nn.conv2d(x, w, [1,1,1,1], padding='SAME')
        return y
    
    def _inverse(self, y):
        print('invertible conv inverse')
        self._init_vars(y)
        w_inv = self._compute_w_inverse(self.L, self.U, self.P, self.log_d, self.sgn_d)
        x = tf.nn.conv2d(y, w_inv, [1,1,1,1], padding='SAME')
        return x
    
    def _forward_log_det_jacobian(self, x):
        self._init_vars(x)
        shape = tf.cast(tf.shape(x), tf.float32)
        return tf.math.reduce_sum(self.log_d)*shape[1]*shape[2]
    
    def _inverse_log_det_jacobian(self, y):
        self._init_vars(y)
        shape = tf.cast(tf.shape(y), tf.float32)
        return -tf.math.reduce_sum(self.log_d)*shape[1]*shape[2]
    