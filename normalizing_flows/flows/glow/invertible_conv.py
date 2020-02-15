import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
from normalizing_flows.flows import Transform

class InvertibleConv(Transform):
    def __init__(self, input_shape=None, name='invertible_1x1_conv', *args, **kwargs):
        self.W = None
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)

    def _initialize(self, input_shape):
        if self.W is None:
            assert input_shape.rank == 4, 'input should be 4-dimensional'
            batch_size, wt, ht, c = input_shape
            ortho_init = tf.initializers.Orthogonal()
            self.W = ortho_init((1,1,c,c))
            #self.W = tf.reshape(tf.eye(c,c), (1,1,c,c))
    
    def _forward(self, x):
        self._init_vars(x)
        y = tf.nn.conv2d(x, self.W, [1,1,1,1], padding='SAME')
        fldj = tf.math.log(tf.math.abs(tf.linalg.det(self.W)))
        return y, tf.broadcast_to(fldj, (x.shape[0],))
    
    def _inverse(self, y):
        self._init_vars(y)
        W_inv = tf.linalg.inv(self.W)
        x = tf.nn.conv2d(y, W_inv, [1,1,1,1], padding='SAME')
        ildj = tf.math.log(tf.math.abs(tf.linalg.det(W_inv)))
        return x, tf.broadcast_to(ildj, (y.shape[0],))
    