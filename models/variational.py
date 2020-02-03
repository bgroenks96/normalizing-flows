import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import flows
from tensorflow.keras import Model
from typing import Callable

def nll_loss(distribution_fn, num_bins=None):
    """
    Implementation of normalized NLL loss function.
    
    distribution_fn : a callable that takes a tf.Tensor and returns a valid
                      TFP Distribution
    num_bins        : number of discretization bins; None for continuous data
    """
    scale_factor = np.log2(num_bins) if num_bins is not None else 1.0
    def nll(y_true, y_pred):
        def _preprocess(self, x):
            if num_bins is not None:
                x += tf.random.uniform(x.shape, 0, 1./num_bins)
            return x
        def log_prob(dist: tfp.distributions.Distribution):
            return dist.log_prob(y_true)
        num_elements = tf.math.reduce_prod(tf.cast(tf.shape(y_true)[1:], dtype=y_true.dtype))
        dist = tfp.layers.DistributionLambda(distribution_fn, log_prob)
        log_probs = tf.math.reduce_sum(dist(y_pred), axis=[i for i in range(1, y_true.shape.rank)])
        nll = -(log_probs - scale_factor*num_elements) / num_elements
        return nll
    return nll

class VariationalModel(Model):
    """
    Extension of Keras Model for supervised, variational inference networks.
    """
    def __init__(self,
                 distribution_fn: Callable[[tf.Tensor], tfp.distributions.Distribution],
                 transform: flows.Transform=flows.Identity(),
                 num_bins=None,
                 *model_args,
                 **model_kwargs):
        """
        distribution_fn : a callable that takes a tf.Tensor and returns a valid
                          TFP Distribution
        transform       : a bijective transform to be applied to the initial density
                          returned by distribution_fn; defaults to Identity (no transform)
        num_bins        : number of discretization bins to use; defaults to None (continuous inputs)
        """
        super().__init__(*model_args, **model_kwargs)
        self.dist_fn = distribution_fn
        self.transform = transform
        self.num_bins = num_bins
        
    def compile(self, **kwargs):
        assert 'loss' not in kwargs, 'NLL loss is automatically supplied by VariationalModel'
        super().compile(loss=nll_loss(self.dist_fn, self.num_bins), **kwargs)
        
    def predict_mean(self, x):
        assert self.transform.has_constant_ldj, 'mean not defined for transforms with variable logdetJ'
        params = self.predict(x)
        dist = self.dist_fn(params)
        y, _ = self.transform.forward(dist.mean())
        return y
    
    def sample(self, x, sample_fn=None):
        params = self.predict(x)
        dist = self.dist_fn(params)
        if sample_fn is not None:
            y, _ = self.transform.forward(sample_fn(dist))
            return y
        else:
            y, _ = self.transform.forward(dist.sample())
            return y
        
    def quantile(self, x, q):
        params = self.predict(x)
        dist = self.dist_fn(params)
        y, _ = self.transform.forward(dist.quantile(q))
        return y
