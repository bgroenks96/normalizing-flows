import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Transform(tf.Module):
    unique_id = 0
    def __init__(self, input_shape: tf.TensorShape=None, requires_init=False, *args, **kwargs):
        self.input_shape = input_shape
        self.requires_init = requires_init
        self.unique_id = Transform.unique_id
        Transform.unique_id += 1
        name = kwargs['name'] if 'name' in kwargs else '{}_{}'.format(type(self).__name__, self.unique_id)
        super().__init__(name=name)
        if input_shape is not None:
            self.initialize(input_shape)
    
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

    def _inverse(self, z, *args, **kwargs):
        """
        Subclass implementation of inverse transform z = f^-1(z')
        """
        raise NotImplementedError('missing implementation of _inverse')
        
    def _regularization_loss(self):
        """
        Optional subclass implementation of regularization loss.
        """
        return tf.constant(0.0, dtype=tf.float32)
    
    def _param_count(self, shape: tf.TensorShape):
        """
        Optional subclass implementation of param_count
        """
        return 0
        
    @tf.function
    def forward(self, z, *args, **kwargs):
        """
        Computes the forward transform z' = f(z)
        Returns z', fldj (forward log det Jacobian)
        """
        assert not self.requires_init or self.input_shape is not None, 'not initialized'
        return self._forward(z, *args, **kwargs)
        
    @tf.function
    def inverse(self, z, *args, **kwargs):
        """
        Computes the inverse transform z = f^-1(z')
        Returns z', ildj (inverse log det Jacobian)
        """
        assert not self.requires_init or self.input_shape is not None, 'not initialized'
        return self._inverse(z, *args, **kwargs)
        
    @tf.function
    def regularization_loss(self):
        """
        Returns the regularization loss for this transform.
        """
        assert not self.requires_init or self.input_shape is not None, 'not initialized'
        return self._regularization_loss()

    def param_count(self, shape: tf.TensorShape):
        """
        Number of parameters for this transform, given input shape
        """
        assert not self.requires_init or self.input_shape is not None, 'not initialized'
        count = self._param_count(shape)
        return count.numpy() if isinstance(count, tf.Tensor) else count
    
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