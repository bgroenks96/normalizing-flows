import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Activation
from normalizing_flows.layers import InstanceNormalization

class PatchDiscriminator(Model):
    """
    Implementation of PatchGAN (Isola et al. 2017) discriminator from CycleGAN (Zhu et al. 2018)
    """
    def __init__(self, input_shape, n_layers=3, n_filters=64, k=3, norm=InstanceNormalization):
        x = Input(input_shape)
        y = x
        for i in range(n_layers):
            f = n_filters * 2**i
            y = Conv2D(f, k, strides=2, padding='same')(y)
            if i > 0:
                y = norm(axis=-1)(y)
            y = LeakyReLU(0.2)(y)
        y = Conv2D(f, k, strides=1, padding='same')(y)
        y = norm(axis=-1)(y)
        y = LeakyReLU(0.2)(y)
        y = Conv2D(1, k, strides=1, padding='same')(y)
        super().__init__(inputs=x, outputs=y)
    