import tensorflow as tf
from tensorflow.keras import layers

class ActNorm(layers.Layer):
    def __init__(self, name='act_norm'):
        super().__init__(name=name)

    def build(self, input_shape):
        self.log_s = self.add_weight(shape=(*[1 for _ in input_shape[:-1]], input_shape[-1]),
                                 initializer='zeros',
                                 name='log_s',
                                 trainable=True)
        self.b = self.add_weight(shape=(*[1 for _ in input_shape[:-1]], input_shape[-1]),
                                 initializer='zeros',
                                 name='b',
                                 trainable=True)

    def call(self, inputs):
        return tf.math.multiply(inputs + self.b, tf.math.exp(self.log_s))
