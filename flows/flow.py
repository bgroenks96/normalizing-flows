import tensorflow as tf
import tensorflow_probability as tfp
from typing import List
from .transform import Transform

class Flow(Transform):
    def __init__(self, steps: List[Transform], input_shape=None, name='flow', *args, **kwargs):
        """
        Constructs a new flow as a sequence of transforms or sub-flows.
        """
        self.steps = steps
        self.num_steps = len(steps)
        # add num_flows alias for legacy code
        self.num_flows = self.num_steps
        super().__init__(*args, input_shape=input_shape, name=name, **kwargs)
    
    @staticmethod
    def uniform(num_flows, transform_init):
        """
        Creates a simple, uniform flow with 'num_flows' steps using the transform_init constructor function.
        transform_init should follow the signature f: i -> Transform, where i is the index of the current step
        in the flow sequence and Transform is a valid transformer instance.
        """
        assert num_flows > 0, "num_flows must be > 0"
        transforms = [transform_init(i) for i in range(num_flows)]
        transform_type = type(transforms[0])
        assert all([transform_type == type(t) for t in transforms]), "All transforms should have the same type for uniform flow"
        return Flow(transforms)
    
    def _initialize(self, input_shape):
        for step in self.steps:
            step.initialize(input_shape)
            input_shape = step._forward_shape(input_shape)

    def _forward(self, z_0, *params: tf.Tensor, return_sequence=False, **kwargs):
        """
        Computes the forward pass of the flow: z_k = f_k . f_k-1 ... f_1(z)

        Tensor shapes:
        z_0    : (batch_size, d)
        params : optional sequence of tensors (batch_size, m_i) where m_i is the number of parameters for flow step i
        """
        assert len(params) == 0 or len(params) == self.num_flows, 'arguments must be provided for all flow steps or none'
        zs = [z_0]
        ldj = 0.0
        for i, step in enumerate(self.steps):
            params_i = [params[i]] if len(params) > 0 else []
            z_i, ldj_i = step.forward(zs[-1], *params_i, **kwargs)
            zs.append(z_i)
            ldj += ldj_i
        return (zs, ldj) if return_sequence else (zs[-1], ldj)
    
    def _inverse(self, z, *params: tf.Tensor, return_sequence=False, **kwargs):
        """
        Computes the inverse pass of the flow: z_0 = f^-1_1 . f^-1_2 ... f^-1_k(z)

        Tensor shapes:
        z_0    : (batch_size, d)
        params : optional sequence of tensors (batch_size, m_i) where m_i is the number of parameters for flow step i
        """
        assert len(params) == 0 or len(params) == self.num_flows, 'arguments must be provided for all flow steps or none'
        zs = [z]
        ldj = 0.0
        for i, step in enumerate(reversed(self.steps)):
            params_i = [params[i]] if len(params) > 0 else []
            z_i, ldj_i = step.inverse(zs[-1], *params_i, **kwargs)
            zs.append(z_i)
            ldj += ldj_i
        return (zs, ldj) if return_sequence else (zs[-1], ldj)
    
    def _regularization_loss(self):
        return tf.math.add_n([t.regularization_loss() for t in self.steps])

    def _param_count(self, shape):
        return tf.math.reduce_sum([t.param_count(shape) for t in self.steps])
