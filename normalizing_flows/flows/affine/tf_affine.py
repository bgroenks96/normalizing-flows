import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from normalizing_flows.flows import Transform

class Affine(Transform):
    """
    DEPRECATED - doesn't seem to work properly; see Planar flow instead
    """
    def __init__(self, input_dims=2, **kwargs):
        super(Affine, self).__init__(input_dims=input_dims, **kwargs)
        self.k = input_dims
        self.scope = f'affine_{self.unique_id}'
        self.V = tf.Variable(np.identity(self.k), name=f'V_{self.unique_id}', dtype=tf.float32)
        self.L = tf.Variable(np.random.uniform(0, 1.0, size=(self.k*(self.k+1) // 2)), name=f'L_{self.unique_id}', dtype=tf.float32)
        self.b = tf.Variable(np.random.normal(0, 0.5, size=(self.k,)), name=f'b_{self.unique_id}', dtype=tf.float32)

    def _forward(self, x):
        affine_bijector = tfp.bijectors.Affine(
            scale_tril=tfp.distributions.fill_triangular(self.L),
            scale_perturb_factor=self.V,
            shift=self.b
        )
        return affine_bijector.forward(x)

    def _inverse(self, y):
        affine_bijector = tfp.bijectors.Affine(
            scale_tril=tfp.distributions.fill_triangular(self.L),
            scale_perturb_factor=self.V,
            shift=self.b
        )
        return affine_bijector.inverse(y)
