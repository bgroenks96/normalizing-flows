import tensorflow as tf
import tensorflow_probability as tfp
from .transform import Transform

class Flow:
    def __init__(self, *steps):
        """
        Constructs a new flow as a sequence of transforms or sub-flows.
        """
        assert all([isinstance(step, Flow) or isinstance(step, Transform) for step in steps]),
               'All steps of flow must be either a sub-type of Transform or another Flow'
        self.steps = steps
        self.num_steps = len(steps)
        # add num_flows alias for legacy code
        self.num_flows = self.num_steps
    
    @staticmethod
    def uniform(self, num_flows, transform_init):
        """
        Creates a simple, uniform flow with 'num_flows' steps using the transform_init constructor function.
        transform_init should follow the signature f: i -> Transform, where i is the index of the current step
        in the flow sequence and Transform is a valid transformer instance.
        """
        assert num_flows > 0, "num_flows must be > 0"
        transforms = [transform_init(i) for i in range(num_flows)]
        transform_type = type(self.transforms[0])
        assert all([transform_type == type(t) for t in self.transforms]), "All transforms should have the same type for uniform flow"
        return Flow(*transforms)

    @tf.function
    def forward(self, z_0, *params: tf.Tensor, return_sequence=False, **kwargs):
        """
        Computes the forward pass of the flow: z_k = f_k . f_k-1 ... f_1(z)

        Tensor shapes:
        z_0    : (batch_size, d)
        params : optional sequence of tensors (batch_size, m_i) where m_i is the number of parameters for flow step i
        """
        assert len(params) == 0 or len(params) == self.num_flows, 'arguments must be provided for all flow steps or none'
        n_flows, n_params = self.num_steps, self.param_count(tf.shape(z_0))
        zs = [z_0]
        ldj = 0.0
        for i, transform in enumerate(self.transforms):
            params_i = [params[i]] if len(params) > 0 else []
            z_k, ldj_k = transform.forward(zs[-1], *params_i)
            ldj += ldj_k
            zs.append(z_k)
        return (zs, ldj) if return_sequence else (zs[-1], ldj)
    
    @tf.function
    def inverse(self, z, *params: tf.Tensor, return_sequence=False, **kwargs):
        """
        Computes the inverse pass of the flow: z_0 = f^-1_1 . f^-1_2 ... f^-1_k(z)

        Tensor shapes:
        z_0    : (batch_size, d)
        params : optional sequence of tensors (batch_size, m_i) where m_i is the number of parameters for flow step i
        """
        assert len(args) == 0 or len(args) == self.num_flows, 'arguments must be provided for all flow steps or none'
        n_flows, n_params = self.num_flows, self.param_count(tf.shape(z))
        args = tf.reshape(args, (-1, n_flows, n_params // n_flows))
        zs = [z]
        ldj = 0.0
        for i, transform in enumerate(self.transforms):
            params_i = [params[i]] if len(params) > 0 else []
            z_j, ldj_j = transform.inverse(zs[-1], args[:,i])
            ldj -= ldj_j
            zs.append(z_j)
        return (zs, ldj) if return_sequence else (zs[-1], ldj)

    def param_count(self, shape):
        return sum([t.param_count(shape) for t in self.transforms])
