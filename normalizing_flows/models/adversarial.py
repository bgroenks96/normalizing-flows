import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Activation
from normalizing_flows.layers import InstanceNormalization

def wasserstein_loss(D, lam=10.):
    """
    Implementation of the Wasserstein loss (Arjovsky et al. 2017) with
    gradient penalty for Lipschitz constraint (Gulrajani et al. 2017).
    
    Returns (D_loss, G_loss); loss functions for the discriminator and
    generator respectively.
    
    D     : discriminator/critic function
    x     : "real" data
    x_gen : "fake" data
    lam   : gradient penalty scalar; defaults to 10, as suggested by the authors
    """
    lam = tf.constant(lam)
    @tf.function
    def D_loss(x_true, x_pred):
        reduction_axes = [i for i in range(1, x_true.shape.rank)]
        d_x = D(x_true)
        d_xg = D(x_pred)
        # compute Wasserstein distance
        wloss = tf.math.reduce_mean(d_xg) - tf.math.reduce_mean(d_x)
        # interpolate xs
        eps = tf.random.uniform(tf.shape(x_true)[:1], minval=0.0, maxval=1.0)
        eps = tf.reshape(eps, (-1,*[1]*(x_true.shape.rank-1)))
        x_i = x_true + eps*(x_pred - x_true)
        dD_dx = tf.gradients(D(x_i), x_i)[0]
        grad_norm = tf.math.sqrt(tf.math.reduce_sum(dD_dx**2, axis=reduction_axes))
        D_loss = wloss + lam*tf.math.reduce_mean((grad_norm - 1.0)**2)
        return D_loss
    @tf.function
    def G_loss(x_true, x_pred):
        return -tf.math.reduce_mean(D(x_pred))
    return D_loss, G_loss

def bce_loss(D, from_logits=True):
    """
    Implementation of traditional GAN discriminator loss with soft/noisy labels.
    """
    @tf.function
    def D_loss(x_true, x_pred):
        pred_real = D(x_true)
        pred_fake = D(x_pred)
        target_real = tf.ones_like(pred_real)
        target_real -= tf.random.normal(tf.shape(target_real), mean=0.1, stddev=0.02)
        target_fake = tf.zeros_like(pred_fake)
        target_fake += tf.random.normal(tf.shape(target_fake), mean=0.1, stddev=0.02)
        loss_real = binary_crossentropy(target_real, pred_real, from_logits=from_logits)
        loss_fake = binary_crossentropy(target_fake, pred_fake, from_logits=from_logits)
        loss = (loss_real + loss_fake)*0.5
        return tf.math.reduce_mean(loss)
    def G_loss(x_true, x_pred):
        # Use discriminator loss with labels inverted
        return D_loss(x_pred, x_true)
    return D_loss, G_loss

class PatchDiscriminator(Model):
    """
    Implementation of PatchGAN (Isola et al. 2017) discriminator from CycleGAN (Zhu et al. 2018)
    """
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
        super().__init__(inputs=x, outputs=y)
    