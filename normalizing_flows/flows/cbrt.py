import tensorflow as tf
import tensorflow_probability as tfp
from normalizing_flows.flows import Transform

class Cbrt(Transform):
    def __init__(self, input_shape=None, name='cbrt', *args, **kwargs):
        super().__init__(*args, input_shape=input_shape, name=name, has_constant_ldj=True, **kwargs)
        
    def _forward(self, x, **kwargs):
        fldj = -tf.math.log(3.0) - 2.0*tf.math.log(x)
        return tf.math.pow(x, 1.0/3.0), fldj
    
    def _inverse(self, y, **kwargs):
        ildj = tf.math.log(3.0) + 2.0*tf.math.log(y)
        return tf.math.pow(y, 3.0), ildj
    
    def _regularization_loss(self):
        return tf.constant(0.0, dtype=tf.float32)
