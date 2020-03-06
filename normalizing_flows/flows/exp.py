import tensorflow as tf
import tensorflow_probability as tfp
from normalizing_flows.flows import Transform

class Exp(Transform):
    def __init__(self, input_shape=None, name='exp', *args, **kwargs):
        super().__init__(*args, input_shape=input_shape, name=name, has_constant_ldj=True, **kwargs)
        
    def _forward(self, x, **kwargs):
        fldj = x
        return tf.math.exp(x), fldj
    
    def _inverse(self, y, **kwargs):
        log_y = tf.math.log(y)
        ildj = -log_y
        return log_y, ildj
    
    def _regularization_loss(self):
        return tf.constant(0.0, dtype=tf.float32)
