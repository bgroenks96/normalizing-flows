import tensorflow as tf
import tensorflow_probability as tfp
from .regularized_bijector import RegularizedBijector

def resnet_glow(hidden_dims=512, kernel_size=3, alpha=1.0E-3):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Lambda
    from tensorflow.keras.regularizers import l2
    def _resnet(c):
        x = Input((None,None,c//2))
        h_1 = Conv2D(hidden_dims, kernel_size, padding='same', kernel_regularizer=l2(alpha))(x)
        h_1 = Activation('relu')(h_1)
        h_1 = BatchNormalization()(h_1)
        h_2 = Conv2D(hidden_dims, 1, padding='same', kernel_regularizer=l2(alpha))(h_1)
        h_2 = Activation('relu')(h_2)
        h_2 = BatchNormalization()(h_2)
        s = Conv2D(c//2, kernel_size, padding='same', kernel_regularizer=l2(alpha), kernel_initializer='zeros')(h_2)
        s = Lambda(lambda x: x+2)(s)
        s = Activation('sigmoid')(s)
        t = Conv2D(c//2, kernel_size, padding='same', kernel_regularizer=l2(alpha), kernel_initializer='zeros')(h_2)
        model = Model(inputs=x, outputs=[s, t])
        # note that loss and optimizer don't matter for individual coupling models
        #model.compile(loss='kld', optimizer='adam')
        return model
    return _resnet

class AffineCoupling(RegularizedBijector):
    def __init__(self, nn_ctor=resnet_glow(),
                 forward_min_event_ndims=1, inverse_min_event_ndims=1,
                 *args, **kwargs):
        super().__init__(forward_min_event_ndims=forward_min_event_ndims,
                         inverse_min_event_ndims=inverse_min_event_ndims,
                         *args, **kwargs)
        self.nn_ctor = nn_ctor
        self.nn = None
        
    def _init_nn(self, x):
        if self.nn is None:
            self.nn = self.nn_ctor(x.shape[-1])
    
    def _forward(self, x):
        self._init_nn(x)
        x_a, x_b = tf.split(x, 2, axis=-1)
        s, t = self.nn(x_b)
        y_a = s*x_a + t
        y_b = x_b
        return tf.concat([y_a, y_b], axis=-1)
    
    def _inverse(self, y):
        self._init_nn(y)
        y_a, y_b = tf.split(y, 2, axis=-1)
        s, t = self.nn(y_b)
        x_a = (y_a - t) / s
        x_b = y_b
        return tf.concat([x_a, x_b], axis=-1)
    
    def _inverse_log_det_jacobian(self, y):
        self._init_nn(y)
        _, y_b = tf.split(y, 2, axis=-1)
        s, _ = self.nn(y_b)
        ildj = -tf.math.reduce_sum(tf.math.log(s), axis=[1,2,3])
        #print(self.name, ildj)
        return tf.expand_dims(ildj, axis=-1)
    
    def _regularization_loss(self):
        assert self.nn is not None, 'bijector not initialized'
        return tf.math.reduce_sum(self.nn.get_losses_for(None))