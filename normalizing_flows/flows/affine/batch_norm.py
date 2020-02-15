import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from normalizing_flows.flows import Transform

class BatchNorm(Transform):
    def __init__(self, input_shape=None, alpha=0., name='batchnorm',
                 *args, **kwargs):    
        """
        Creates a new batch normalization (batchnorm) transform.
        """
        self.alpha = alpha
        self.log_s = None
        self.b = None
        self.init = False
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)
        
    def _initialize(self, input_shape):
        if not self.init:
            mus = tf.random.normal([1 for _ in range(input_shape.rank-1)] + [input_shape[-1]], mean=0.0, stddev=0.1)
            log_sigmas = tf.random.normal([1 for _ in range(input_shape.rank-1)] + [input_shape[-1]], mean=0.0, stddev=0.1)
            self.log_s = tf.Variable(log_sigmas, name=f'{self.name}/log_s')
            self.b = tf.Variable(mus, name=f'{self.name}/b')
            self.init = True
            
    def _standardize(self, x):
        # assign initial values based on mean/stdev of first batch
        input_shape = x.shape
        mus = tf.math.reduce_mean(x, axis=[i for i in range(input_shape.rank-1)], keepdims=True)
        if sum(input_shape[1:-1]) > 2:
            sigmas = tf.math.reduce_std(x, axis=[i for i in range(input_shape.rank-1)], keepdims=True)
        else:
            # if all non-channel dimensions have only one element, set variances to one to avoid inf values
            sigmas = tf.ones(input_shape)
        return (x - mus) / (sigmas+1.0E-6)

    def _forward(self, x, *args, **kwargs):
        x = self._standardize(x)
        y = tf.math.exp(self.log_s)*x + self.b
        fldj = tf.math.reduce_sum(self.log_s)*np.prod(y.shape[1:-1])
        return y, fldj*tf.ones(tf.shape(x)[:1])
        
    def _inverse(self, y, *args, **kwargs):
        y = self._standardize(y)
        x = tf.math.exp(-self.log_s)*(y - self.b)
        ildj = -tf.math.reduce_sum(self.log_s)*np.prod(y.shape[1:-1])
        return x, ildj*tf.ones(tf.shape(y)[:1])
    
    def _regularization_loss(self):
        return self.alpha*tf.math.reduce_sum(self.log_s**2)