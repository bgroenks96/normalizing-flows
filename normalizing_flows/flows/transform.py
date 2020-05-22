import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Transform(tf.Module):
    def __init__(self,
                 input_shape: tf.TensorShape=None,
                 requires_init=False,
                 has_constant_ldj=False,
                 *args, **kwargs):
        self.input_shape = input_shape
        self.requires_init = requires_init
        self.has_constant_ldj = has_constant_ldj
        name = kwargs['name'] if 'name' in kwargs else type(self).__name__
        super().__init__(name=name)
        if input_shape is not None:
            self.initialize(input_shape)

    def __call__(self, z, *args, **kwargs):
        return self.forward(z, *args, **kwargs)

    def _initialize(self, input_shape):
        """
        Optional subclass implementation for variable initialization.
        """
        pass

    def _forward(self, z, *args, **kwargs):
        """
        Subclass implementation of forward transform z' = f(z)
        """
        raise NotImplementedError('missing implementation of _forward')

    def _inverse(self, z, *args, **kwargs):
        """
        Subclass implementation of inverse transform z = f^-1(z')
        """
        raise NotImplementedError('missing implementation of _inverse')

    def _forward_shape(self, shape: tf.TensorShape):
        """
        Permutes shape according to the forward transform.
        """
        return shape

    def _inverse_shape(self, shape: tf.TensorShape):
        """
        Permutes shape according to the inverse transform.
        """
        return shape

    def _regularization_loss(self):
        """
        Optional subclass implementation of regularization loss.
        """
        return tf.constant(0.0, dtype=tf.float32)

    def forward(self, z, *args, **kwargs):
        """
        Computes the forward transform z' = f(z)
        Returns z', fldj (forward log det Jacobian)
        """
        assert not self.requires_init or self.input_shape is not None, 'not initialized'
        return self._forward(z, *args, **kwargs)

    def inverse(self, z, *args, **kwargs):
        """
        Computes the inverse transform z = f^-1(z')
        Returns z', ildj (inverse log det Jacobian)
        """
        assert not self.requires_init or self.input_shape is not None, 'not initialized'
        return self._inverse(z, *args, **kwargs)

    def regularization_loss(self):
        """
        Returns the regularization loss for this transform.
        """
        assert not self.requires_init or self.input_shape is not None, 'not initialized'
        return self._regularization_loss()

    def initialize(self, input_shape: tf.TensorShape):
        """
        Initializes variables and constants for this transform.
        This step may not be required for all implementations.
        """
        assert input_shape is not None, 'input shape must be provided'
        input_shape = tf.TensorShape(input_shape)
        self.input_shape = input_shape
        self._initialize(input_shape)

    def is_initialized(self):
        return self.input_shape is not None

class AmortizedTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _param_count(self, shape: tf.TensorShape):
        """
        Optional subclass implementation of param_count
        """
        return 0

    def _create_variables(self, shape: tf.TensorShape, initializer=None, **var_kwargs):
        """
        Optional subclass implementation of create_variables
        """
        if initializer is None:
            initializer = lambda shape: tf.random.uniform(shape)
        return [tf.Variable(initializer((1,self.param_count(shape))), **var_kwargs)]

    def param_count(self, shape: tf.TensorShape):
        """
        Number of parameters for this transform, given input shape
        """
        count = self._param_count(shape)
        return count.numpy() if isinstance(count, tf.Tensor) else count

    def create_variables(self, shape: tf.TensorShape, initializer=None, **var_kwargs):
        """
        Convenience function for initializing variables that may be used for this amortized transform,
        i.e. creates one or more variables with a total of param_count values. Note that this is not
        generally necessary, but may be useful for some applications.
        """
        return self._create_variables(shape, **var_kwargs)
