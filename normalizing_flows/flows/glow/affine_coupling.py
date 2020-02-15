import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from normalizing_flows.flows import Transform

def coupling_nn_glow(min_filters=32, max_filters=512, kernel_size=3, num_blocks=1, alpha=1.0E-5, epsilon=1.0E-4):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Lambda, add
    from tensorflow.keras.regularizers import l2
    from normalizing_flows.layers import ActNorm
    def _resnet_block(x, num_filters):
        h = Conv2D(num_filters, kernel_size, padding='same', kernel_regularizer=l2(alpha))(x)
        h = ActNorm()(h)
        h = Activation('relu')(h)
        h = Conv2D(num_filters, kernel_size, padding='same', kernel_regularizer=l2(alpha))(h)
        h = ActNorm()(h)
        h = add([x, h])
        h = Activation('relu')(h)
        return h
    def f(i, c, log_scale: tf.Variable):
        num_filters = np.minimum(min_filters*2**i, max_filters)
        x = Input((None,None,c//2))
        h = Conv2D(num_filters, kernel_size, padding='same', kernel_regularizer=l2(alpha))(x)
        h = Activation('relu')(h)
        for i in range(num_blocks):
            h = _resnet_block(h, num_filters)
        s = Conv2D(c//2, kernel_size, padding='same', kernel_regularizer=l2(alpha), kernel_initializer='zeros')(h)
        s = Lambda(lambda x: x*tf.math.exp(log_scale))(s)
        s = Activation(lambda x: tf.nn.sigmoid(x)+0.1)(s)
        t = Conv2D(c//2, kernel_size, padding='same', kernel_initializer='zeros')(h)
        model = Model(inputs=x, outputs=[s, t])
        return model
    return f

@tf.function
def affine(x, s, t, inverse=tf.constant(False)):
    ldj = tf.math.reduce_sum(tf.math.log(s), axis=[1,2,3])
    if inverse:
        y = x / s - t
        ldj *= -1
    else:
        y = (x + t)*s
    return y, ldj

class AffineCoupling(Transform):
    def __init__(self, layer, input_shape=None, nn_ctor=coupling_nn_glow(), name='affine_coupling', *args, **kwargs):
        self.layer = layer
        self.nn_ctor = nn_ctor
        self.nn = None
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)
        
    def _initialize(self, input_shape):
        if self.nn is None:
            self.log_scale = tf.Variable(tf.zeros((1,1,1,input_shape[-1]//2)), dtype=tf.float32, name=f'{self.name}/log_scale')
            self.nn = self.nn_ctor(self.layer, input_shape[-1], self.log_scale)
    
    def _forward(self, x, **kwargs):
        x_a, x_b = tf.split(x, 2, axis=-1)
        s, t = self.nn(x_b)
        y_a, fldj = affine(x_a, s, t)
        y_b = x_b
        return tf.concat([y_a, y_b], axis=-1), fldj
    
    def _inverse(self, y, **kwargs):
        y_a, y_b = tf.split(y, 2, axis=-1)
        s, t = self.nn(y_b)
        x_a, ildj = affine(y_a, s, t, inverse=tf.constant(True))
        x_b = y_b
        return tf.concat([x_a, x_b], axis=-1), ildj
    
    def _regularization_loss(self):
        assert self.nn is not None, 'bijector not initialized'
        return tf.math.reduce_sum(self.nn.get_losses_for(None))