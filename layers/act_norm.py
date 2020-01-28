import tensorflow as tf
from tensorflow.keras import layers

class ActNorm(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.w = self.add_weight(shape=(*[1 for _ in input_shape[:-1]], input_shape[-1]),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(*[1 for _ in input_shape[:-1]], input_shape[-1]),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.math.multiply(inputs + self.b, self.w)
