import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .utils import tf_init_var

class ActNorm(tfp.bijectors.Bijector):
    def __init__(self, event_ndims=1, init_from_data=False, *args, **kwargs):
        """
        Creates a new activation normalization (actnorm) bijector.
        """
        super().__init__(forward_min_event_ndims=event_ndims,
                         inverse_min_event_ndims=event_ndims,
                         *args, **kwargs)
        self.event_ndims = event_ndims
        self.init_from_data = init_from_data
        self.log_s = tf_init_var(event_ndims=event_ndims, constraint=lambda logs: tf.math.abs(logs))
        self.b = tf_init_var(event_ndims=event_ndims)
        self.init = False
        
    def _init_vars(self, x):
        if not self.init and self.init_from_data:
            input_shape = tf.shape(x)
            # assign initial values based on mean/stdev of first batch
            mus = tf.math.reduce_mean(x, axis=[i for i in range(self.event_ndims)], keepdims=True)
            sigmas = tf.math.reduce_std(x, axis=[i for i in range(self.event_ndims)], keepdims=True)
            self.log_s.assign(-tf.math.log(sigmas))
            self.b.assign(-mus)
            self.init = True
        elif not self.init:
            input_shape = tf.shape(x)
            mus = tf.random.normal([1 for _ in range(self.event_ndims)] + [input_shape[-1]], mean=0.0, stddev=0.1)
            log_sigmas = tf.random.normal([1 for _ in range(self.event_ndims)] + [input_shape[-1]], mean=1.0, stddev=0.1)
            self.log_s.assign(log_sigmas)
            self.b.assign(mus)
            self.init = True
    
    def _forward(self, x):
        self._init_vars(x)
        return tf.math.exp(self.log_s)*x + self.b
        
    def _inverse(self, y):
        self._init_vars(y)
        return tf.math.exp(-self.log_s)*(y - self.b)
        
    def _forward_log_det_jacobian(self, x):
        self._init_vars(x)
        factor = self._log_det_scale_factor(x)
        return factor*tf.math.reduce_sum(self.log_s)
    
    def _inverse_log_det_jacobian(self, y):
        self._init_vars(y)
        factor = self._log_det_scale_factor(y)
        return -factor*tf.math.reduce_sum(self.log_s)
    
    def _log_det_scale_factor(self, x):
        shape = tf.shape(x)
        if len(shape) <= 2:
            return 1.0
        elif len(shape) == 3:
            return tf.cast(shape[1], dtype=x.dtype)
        elif len(shape) == 4:
            return tf.cast(shape[1]*shape[2], dtype=x.dtype)
        else:
            raise Exception('unsupported shape {}'.format(shape))