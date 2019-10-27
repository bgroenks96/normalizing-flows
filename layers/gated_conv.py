import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2

class GatedConv2D(layers.Layer):
    def __init__(self, filters, kernel, strides=1, dilation_rate=1, padding='same', activation='linear'):
        super(GatedConv2D, self).__init__()
        self.h = layers.Conv2D(filters, kernel, strides=strides, padding=padding, dilation_rate=dilation_rate,
                               kernel_regularizer=l2(1.0E-5))
        self.g = layers.Conv2D(filters, kernel, strides=strides, padding=padding, dilation_rate=dilation_rate)
        self.a = layers.Activation(activation)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        h = self.a(self.h(x))
        return h*self.sigmoid(self.g(x))

class GatedConv2DTranspose(layers.Layer):
    def __init__(self, filters, kernel, strides=1, dilation_rate=1, padding='same', activation='linear'):
        super(GatedConv2DTranspose, self).__init__()
        self.h = layers.Conv2DTranspose(filters, kernel, strides=strides, padding=padding, dilation_rate=dilation_rate,
                                        kernel_regularizer=l2(1.0E-5))
        self.g = layers.Conv2DTranspose(filters, kernel, strides=strides, padding=padding, dilation_rate=dilation_rate)
        self.a = layers.Activation(activation)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        h = self.a(self.h(x))
        return h*self.sigmoid(self.g(x))
