import tensorflow as tf
from flows import Transform

class Upsample(Transform):
    """
    Implements a nearest neighbors upsampling bijector.
    """
    def __init__(self, factor=2, input_shape=None, name='upsample',
                 *args, **kwargs):
        self.factor = factor
        super().__init__(*args,
                         input_shape=input_shape,
                         requires_init=True,
                         has_constant_ldj=True,
                         name=name, **kwargs)
        
    def _forward_shape(self, input_shape):
        return tf.TensorShape((input_shape[0],
                               input_shape[1]*self.factor,
                               input_shape[2]*self.factor,
                               input_shape[3]))
    
    def _inverse_shape(self, input_shape):
        return tf.TensorShape((input_shape[0],
                               input_shape[1]//self.factor,
                               input_shape[2]//self.factor,
                               input_shape[3]))
        
    def _forward(self, x):
        return tf.image.resize(x, (x.shape[1]*self.factor, x.shape[2]*self.factor),
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), 0.0
        
    def _inverse(self, y):
        return tf.image.resize(x, (x.shape[1]//self.factor, x.shape[2]//self.factor),
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), 0.0