import tensorflow as tf
import tensorflow_probability as tfp
from normalizing_flows.flows import Transform

class Softplus(Transform):
    def __init__(self, input_shape=None, name='softplus', *args, **kwargs):
        super().__init__(*args, input_shape=input_shape, name=name, has_constant_ldj=True, **kwargs)
        
    def _forward(self, x, **kwargs):
        fldj = -tf.math.reduce_sum(tf.math.softplus(-x), axis=[i for i in range(1,x.shape.rank)])
        return tf.math.softplus(x), fldj
    
    def _inverse(self, y, **kwargs):
        ildj = -tf.math.reduce_sum(tf.math.log(-tf.math.expm1(-y)), axis=[i for i in range(1,y.shape.rank)])
        return tfp.math.softplus_inverse(y), ildj
    
    def _regularization_loss(self):
        return tf.constant(0.0, dtype=tf.float32)
