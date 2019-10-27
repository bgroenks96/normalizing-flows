import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class BaseTransform(tf.Module):
    unique_id = 0
    def __init__(self, **kwargs):
        self.unique_id = BaseTransform.unique_id
        BaseTransform.unique_id += 1
        name = kwargs['name'] if 'name' in kwargs else '{}_{}'.format(type(self).__name__, self.unique_id)
        super(BaseTransform, self).__init__(name=name)

    def forward(self, z, *args):
        raise NotImplementedError('missing implementation of forward')

    def inverse(self, x, *args):
        raise NotImplementedError('missing implementation of inverse')

    def param_count(self, d):
        """
        Number of parameters for this transform, given number of z dims, d
        """
        raise NotImplementedError('missing implementation of param_count')
