import tensorflow as tf
import tensorflow_probability as tfp

class RegularizedBijector(tfp.bijectors.Bijector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _regularization_loss(self):
        return tf.constant(0., dtype=tf.float32)
