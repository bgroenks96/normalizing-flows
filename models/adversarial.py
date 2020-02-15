import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Activation
from layers import InstanceNormalization

def loss(pred_real, pred_fake):
    target_real = tf.ones_like(pred_real)
    target_fake = tf.zeros_like(pred_fake)
    loss_real = binary_crossentropy(target_real, pred_real)
    loss_fake = binary_crossentropy(target_fake, pred_fake)
    loss = (loss_real + loss_fake)*0.5
    return tf.math.reduce_mean(loss, axis=[i for i in range(1, loss.shape.rank)])

class PatchDiscriminator(Model):
    def __init__(self, input_shape, n_layers=3, n_filters=64, k=3, norm=InstanceNormalization):
        def bce(y_true, y_pred):
            return binary_crossentropy(y_true, y_pred)
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
        pred = Activation('sigmoid', name='sigmoid')(y)
        super().__init__(inputs=x, outputs=y)
    