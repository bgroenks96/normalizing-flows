import tensorflow as tf
from tensorflow.keras import layers

class ActNorm(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.log_s = self.add_weight(shape=(*[1 for _ in input_shape[:-1]], input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        self.b = self.add_weight(shape=(*[1 for _ in input_shape[:-1]], input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.math.multiply(inputs + self.b, tf.math.exp(self.log_s))
