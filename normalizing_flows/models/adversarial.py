import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Activation
from normalizing_flows.layers import InstanceNormalization

@tf.function
def wasserstein_loss(D, x, x_gen, lam=10):
    """
    Implementation of the Wasserstein loss (Arjovsky et al. 2017) with
    gradient penalty for Lipschitz constraint (Gulrajani et al. 2017).
    Returns (D_loss, G_loss); losses for the discriminator D and
    Generator G respectively.
    
    D : discriminator/critic function
    x : "real" data
    x_gen : "fake" data
    lam : gradient penalty scalar; defaults to 10, as suggested by the authors
    """
    reduction_axes = [i for i in range(1, x.shape.rank)]
    d_x = D(x)
    d_xg = D(x_gen)
    # compute Wasserstein loss
    wloss = tf.math.reduce_mean(d_x) - tf.math.reduce_mean(d_xg)
    # interpolate xs
    eps = tf.reshape(tf.random.uniform(tf.shape(x)[:1]), (-1,*[1]*(x.shape.rank-1)))
    x_i = x + eps*(x_gen - x)
    dD_dx = tf.gradients(D(x_i), x_i)[0]
    grad_norm = tf.math.sqrt(tf.math.reduce_sum(dD_dx**2, axis=reduction_axes))
    D_loss = wloss + lam*(grad_norm - 1.0)**2
    G_loss = -wloss
    return D_loss, G_loss

@tf.function
def bce_loss(pred_real, pred_fake):
    """
    Implementation of traditional GAN discriminator loss with soft/noisy labels.
    """
    target_real = tf.ones_like(pred_real)
    target_real -= tf.random.normal(tf.shape(target_real), mean=0.1, stddev=0.02)
    target_fake = tf.zeros_like(pred_fake)
    target_fake += tf.random.normal(tf.shape(target_fake), mean=0.1, stddev=0.02)
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
    