import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class BaseTransform(tfp.bijectors.Bijector, tf.Module):
    unique_id = 0
    def __init__(self, **kwargs):
        min_event_ndims = kwargs['min_event_ndims'] if 'min_event_ndims' in kwargs else 1
        validate_args = kwargs['validate_args'] if 'validate_args' in kwargs else False
        name = kwargs['name'] if 'name' in kwargs else type(self).__name__
        super(BaseTransform, self).__init__(
            forward_min_event_ndims=min_event_ndims,
            inverse_min_event_ndims=min_event_ndims,
            validate_args=validate_args,
            name=name)
        self.kwargs = kwargs
        self.unique_id = BaseTransform.unique_id
        BaseTransform.unique_id += 1

    def _forward(self, x):
        raise NotImplementedError('missing implementation of _forward')

    def _inverse(self, y):
        raise NotImplementedError('missing implementation of _inverse')

    def _inverse_log_det_jacobian(self, y):
        raise NotImplementedError('missing implementation of _inverse_log_det_jacobian')

class AmortizedTransform(BaseTransform):
    def __init__(self, param_count, **kwargs):
        super(AmortizedTransform, self).__init__(**kwargs)
        self.param_count = param_count

    def _amortize(self, args: tf.Tensor):
        raise NotImplementedError('missing implementation of _amortize')

    def amortize(self, args: tf.Tensor):
        """
        Function that accepts amortized parameters.
        """
        return self._amortize(args)