import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
from .utils import tf_init_var

class InvertibleConv(tfp.bijectors.Bijector):
    def __init__(self, name='invertible_1x1_conv', forward_min_event_ndims=1, inverse_min_event_ndims=1,
                 *args, **kwargs):
        super().__init__(*args,
                         forward_min_event_ndims=forward_min_event_ndims,
                         inverse_min_event_ndims=inverse_min_event_ndims,
                         name=name, **kwargs)
        self.W = None

    def _init_vars(self, x):
        if self.W is None:
            input_shape = x.shape
            assert input_shape.rank == 4, 'input should be 4-dimensional'
            batch_size, wt, ht, c = input_shape
            ortho_init = tf.initializers.Orthogonal()
            self.W = ortho_init((1,1,c,c))
            #self.W = tf.reshape(tf.eye(c,c), (1,1,c,c))
    
    def _forward(self, x):
        self._init_vars(x)
        y = tf.nn.conv2d(x, self.W, [1,1,1,1], padding='SAME')
        return y
    
    def _inverse(self, y):
        self._init_vars(y)
        x = tf.nn.conv2d(y, tf.linalg.inv(self.W), [1,1,1,1], padding='SAME')
        return x
    
    def _forward_log_det_jacobian(self, x):
        self._init_vars(x)
        det = tf.linalg.det(self.W)
        fldj = tf.math.log(tf.math.abs(det))
        return tf.squeeze(tf.broadcast_to(fldj, (x.shape[0],1)))

    def _inverse_log_det_jacobian(self, x):
        self._init_vars(x)
        det = tf.linalg.det(tf.linalg.inv(self.W))
        ildj = tf.math.log(tf.math.abs(det))
        return tf.broadcast_to(ildj, (x.shape[0],1))
    