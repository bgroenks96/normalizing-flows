import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
from flows import Transform

class InvertibleConv(Transform):
    def __init__(self, input_shape=None, alpha=1.0, name='invertible_1x1_conv', *args, **kwargs):
        self.alpha = alpha
        self.init = False
        super().__init__(*args, input_shape=input_shape, name=name, requires_init=True, **kwargs)

    def _initialize(self, input_shape):
        if not self.init:
            assert input_shape.rank == 4, f'input should be 4-dimensional, got {input_shape}'
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
            self.P = tf.Variable(np.expand_dims(p, axis=0).astype(np.float32), trainable=False, name=f'{self.name}/P')
            self.L = tf.Variable(np.expand_dims(l, axis=0).astype(np.float32), name=f'{self.name}/L')
            self.U = tf.Variable(np.expand_dims(u, axis=0).astype(np.float32), name=f'{self.name}/U')
            self.log_d = tf.Variable(np.expand_dims(log_d, axis=0).astype(np.float32), name=f'{self.name}/log_d')
            self.sgn_d = tf.Variable(np.expand_dims(sgn_d, axis=0).astype(np.float32), trainable=False, name=f'{self.name}/sgn_d')
            self.tril_mask = tf.constant(np.tril(np.ones((1,c,c)), k=-1), dtype=tf.float32)
            self.init = True
    
    def _compute_w(self, l, u, p, log_d, sgn_d):
        d = tf.linalg.diag(tf.math.exp(log_d)*sgn_d)
        l = self.tril_mask*l + tf.eye(self.input_shape[-1])
        u = tf.transpose(self.tril_mask, [0,2,1])*u + d
        w = tf.linalg.matmul(p, tf.linalg.matmul(l, u))
        return tf.expand_dims(w, axis=0) # (1,1,c,c)
    
    def _compute_w_inverse(self, l, u, p, log_d, sgn_d):
        d = tf.linalg.diag(tf.math.exp(log_d)*sgn_d)
        tf.debugging.assert_all_finite(l, 'L has nan/inf values')
        l_inv = tf.linalg.inv(self.tril_mask*l + tf.eye(self.input_shape[-1]))
        u_inv = tf.linalg.inv(tf.transpose(self.tril_mask, [0,2,1])*u + d)
        p_inv = tf.linalg.inv(p)
        w_inv = tf.linalg.matmul(u_inv, tf.linalg.matmul(l_inv, p_inv))
        return tf.expand_dims(w_inv, axis=0) # (1,1,c,c)
    
    def _forward(self, x):
        w = self._compute_w(self.L, self.U, self.P, self.log_d, self.sgn_d)
        y = tf.nn.conv2d(x, w, [1,1,1,1], padding='SAME')
        fldj = tf.math.reduce_sum(self.log_d)*np.prod(x.shape[1:-1])
        return y, tf.broadcast_to(fldj, (x.shape[0],))
    
    def _inverse(self, y):
        w_inv = self._compute_w_inverse(self.L, self.U, self.P, self.log_d, self.sgn_d)
        x = tf.nn.conv2d(y, w_inv, [1,1,1,1], padding='SAME')
        ildj = -tf.math.reduce_sum(self.log_d)*np.prod(y.shape[1:-1])
        return x, tf.broadcast_to(ildj, (y.shape[0],))
    
    def _regularization_loss(self):
        return self.alpha*tf.math.reduce_sum(self.log_d**2)
    