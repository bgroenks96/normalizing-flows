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
        self.pre = kwargs['pre'] if 'pre' in kwargs else None
        self.kwargs = kwargs
        self.unique_id = BaseTransform.unique_id
        BaseTransform.unique_id += 1

    def __call__(self, input_transform):
        kwargs = dict(self.kwargs)
        kwargs['pre'] = input_transform
        return self.__class__(**kwargs)

    def _forward(self, x):
        raise NotImplementedError('missing implementation of _forward')

    def _inverse(self, y):
        raise NotImplementedError('missing implementation of _inverse')

    def _inverse_log_det_jacobian(self, y):
        raise NotImplementedError('missing implementation of _inverse_log_det_jacobian')

class Affine(BaseTransform):
    def __init__(self, input_dims=2, hidden_dims=2, **kwargs):
        super(Affine, self).__init__(input_dims=input_dims, hidden_dims=hidden_dims, **kwargs)
        self.k = input_dims
        self.r = hidden_dims
        self.scope = f'affine_{self.unique_id}'
        self.V = tf.Variable(np.random.normal(0, 2, size=(self.k, self.r)), name=f'V_{self.unique_id}', dtype=tf.float32)
        self.L = tf.Variable(np.random.normal(0, 2, size=(self.k*(self.k+1) // 2)), name=f'L_{self.unique_id}', dtype=tf.float32)
        self.b = tf.Variable(np.random.normal(0, 2, size=(self.k,)), name=f'b_{self.unique_id}', dtype=tf.float32)
        # set up affine bijector
        self.affine_bijector = tfp.bijectors.Affine(
            scale_tril=tfp.distributions.fill_triangular(self.L),
            scale_perturb_factor=self.V,
            shift=self.b
        )

    def _forward(self, x):
        return self.affine_bijector.forward(x)

    def _inverse(self, y):
        return self.affine_bijector.inverse(y)

    def _inverse_log_det_jacobian(self, y):
        return self.affine_bijector.inverse_log_det_jacobian(y, self.inverse_min_event_ndims)

class PReLU(BaseTransform):
    def __init__(self, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.scope = f'prelu_{self.unique_id}'
        self.alpha = tf.Variable(0.01, name='alpha', dtype=tf.float32)

    def _forward(self, x):
        alpha = tf.abs(self.alpha)
        return tf.where(x >= 0, x, alpha*x)

    def _inverse(self, y):
        alpha = tf.abs(self.alpha)
        return tf.where(y >= 0, y, 1.0 / alpha * y)

    def _inverse_log_det_jacobian(self, y):
        I = tf.ones_like(y)
        alpha = tf.abs(self.alpha)
        inv_jacobian = tf.where(y >= 0, I, I * 1.0 / alpha)
        return tf.reduce_sum(tf.math.log(tf.math.abs(inv_jacobian)))
