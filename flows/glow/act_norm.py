import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .regularized_bijector import RegularizedBijector

class ActNorm(RegularizedBijector):
    def __init__(self, alpha=0.1, init_from_data=False, name='act_norm',
                 forward_min_event_ndims=3, inverse_min_event_ndims=3,
                 *args, **kwargs):    
        """
        Creates a new activation normalization (actnorm) bijector.
        """
        super().__init__(forward_min_event_ndims=forward_min_event_ndims,
                         inverse_min_event_ndims=inverse_min_event_ndims,
                         name=name,
                         *args, **kwargs)
        self.alpha = alpha
        self.init_from_data = init_from_data
        self.init = False
        
    def _init_vars(self, x):
        if not self.init and self.init_from_data:
            input_shape = x.shape
            # assign initial values based on mean/stdev of first batch
            mus = tf.math.reduce_mean(x, axis=[i for i in range(input_shape.rank-1)], keepdims=True)
            sigmas = tf.math.reduce_std(x, axis=[i for i in range(input_shape.rank-1)], keepdims=True)
            self.log_s = tf.Variable(-tf.math.log(sigmas), name=f'{self.name}/log_s')
            self.b = tf.Variable(-mus, name=f'{self.name}/b')
            self.init = True
        elif not self.init:
            input_shape = x.shape
            mus = tf.random.normal([1 for _ in range(input_shape.rank-1)] + [input_shape[-1]], mean=0.0, stddev=0.1)
            log_sigmas = tf.random.normal([1 for _ in range(input_shape.rank-1)] + [input_shape[-1]], mean=1.0, stddev=0.1)
            self.log_s = tf.Variable(log_sigmas, name=f'{self.name}/log_s')
            self.b = tf.Variable(mus, name=f'{self.name}/b')
            self.init = True
    
    def _inverse(self, x):
        self._init_vars(x)
        return tf.math.exp(self.log_s)*x + self.b
        
    def _forward(self, y):
        self._init_vars(y)
        return tf.math.exp(-self.log_s)*(y - self.b)
        
    def _inverse_log_det_jacobian(self, x):
        self._init_vars(x)
        ildj = tf.math.reduce_sum(self.log_s)
        return tf.broadcast_to(ildj, (x.shape[0],))
    
    def _forward_log_det_jacobian(self, y):
        return -self._inverse_log_det_jacobian(y)
    
    def _regularization_loss(self):
        return self.alpha*tf.math.reduce_sum(self.log_s**2)