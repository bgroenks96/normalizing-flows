import tensorflow as tf
import tensorflow_probability as tfp
from flows import Transform
from . import Parameterize

class Split(Transform):
    """
    Implementation of the 'split' transform, which factors out half of the channel dimensions
    for parameterization under the prior.
    """
    def __init__(self,
                 parameterize: Parameterize,
                 input_shape=None,
                 split_axis=-1, 
                 name='parameterize',
                 *args, **kwargs):
        """
        parameterize : an implementation of Parameterize that parameterizes the latent variables
                       as a distribution in the same family as the prior.
        split_axis   : which axis to split along
        """
        self.parameterize = parameterize
        self.split_axis = split_axis
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)
        
    def _forward_shape(self, input_shape):
        axis = self.split_axis % input_shape.rank
        new_size = input_shape[axis] // 2
        new_shape = tf.TensorShape((*input_shape[:axis], new_size, *input_shape[axis+1:]))
        return new_shape
    
    def _inverse_shape(self, input_shape):
        axis = self.split_axis % input_shape.rank
        new_size = input_shape[axis] * 2
        new_shape = tf.TensorShape((*input_shape[:axis], new_size, *input_shape[axis+1:]))
        return new_shape
        
    def _initialize(self, input_shape):
        split_shape = self._forward_shape(input_shape)
        self.parameterize.initialize(split_shape)
        
    def _forward(self, x, **kwargs):
        x1, x2 = tf.split(x, 2, axis=self.split_axis)
        z2, fldj = self.parameterize.forward(x1, x2, **kwargs)
        return (x1, z2), fldj
        
    def _inverse(self, x1, z2, **kwargs):
        x2, ildj = self.parameterize.inverse(x1, z2)
        x = tf.concat((x1, x2), axis=self.split_axis, **kwargs)
        return x, ildj