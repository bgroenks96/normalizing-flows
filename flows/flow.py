import tensorflow as tf
import tensorflow_probability as tfp
from .base_transform import BaseTransform

class Flow():
    def __init__(self, num_flows, transform_init):
        assert num_flows > 0, "num_flows must be > 0"
        self.num_flows = num_flows
        self.transforms = [transform_init(i) for i in range(num_flows)]
        self.type = type(self.transforms[0])
        assert all([self.type == type(t) for t in self.transforms]), "All transforms should have the same type"

    @tf.function
    def forward(self, z_0, args: tf.Tensor):
        """
        Computes the forward pass of the flow: z_k = f_k . f_k-1 ... f_1(z)

        Tensor shapes:
        z_0    : (batch_size, d)
        args : (batch_size, m) where m is equal to the total number of parameters for all flows
        """
        n_flows, n_params = self.num_flows, self.param_count(tf.shape(z_0)[1])
        args = tf.reshape(args, (-1, n_flows, n_params // n_flows))
        zs = [z_0]
        ldj = 0.0
        for i, transform in enumerate(self.transforms):
            z_k, ldj_k = transform.forward(zs[-1], args[:,i])
            ldj += ldj_k
            zs.append(z_k)
        return zs, ldj

    def param_count(self, d):
        return sum([t.param_count(d) for t in self.transforms])
