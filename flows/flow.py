import tensorflow as tf
import tensorflow_probability as tfp
from .base_transform import BaseTransform

class Flow():
    def __init__(self, transform: BaseTransform):
        self.transforms = [transform]

    def __call__(self, x):
        assert isinstance(x, BaseTransform)
        self.transforms.append(x)
        return self

    @tf.function
    def forward(self, z, args: tf.Tensor):
        """
        Computes the forward pass of the flow: z_k = f_k . f_k-1 ... f_1(z)

        Tensor shapes:
        z    : (batch_size, d)
        args : (batch_size, m) where m is equal to the total number of parameters for all flows
        """
        n_flows, n_params = self.num_flows(), self.param_count(tf.shape(z)[1])
        args = tf.reshape(args, (-1, n_flows, n_params // n_flows))
        z_k = z
        ldj = 0.0
        for i, transform in enumerate(self.transforms):
            z_k, ldj_k = transform.forward(z_k, args[:,i])
            ldj += ldj_k
        return z_k, ldj

    def param_count(self, d):
        return sum([t.param_count(d) for t in self.transforms])

    def num_flows(self):
        return len(self.transforms)
