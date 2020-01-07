import tensorflow as tf
import tensorflow_probability as tfp

def resnet_glow(hidden_dims=512, kernel_size=3, alpha=1.0E-4):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Conv2D, add
    from tensorflow.keras.regularizers import l2
    def _resnet(c):
        x = Input((None,None,c//2))
        h_1 = Conv2D(hidden_dims, kernel_size, activation='relu', padding='same', kernel_regularizer=l2(alpha))(x)
        h_2 = Conv2D(hidden_dims, 1, activation='relu', padding='same', kernel_regularizer=l2(alpha))(h_1)
        h_2 = add([h_2, h_1])
        log_s = Conv2D(c//2, kernel_size, padding='same', kernel_regularizer=l2(alpha), kernel_initializer='zeros')(h_2)
        t = Conv2D(c//2, kernel_size, padding='same', kernel_regularizer=l2(alpha), kernel_initializer='zeros')(h_2)
        model = Model(inputs=x, outputs=[log_s, t])
        return model
    return _resnet

class AffineCoupling(tfp.bijectors.Bijector):
    def __init__(self, nn_ctor=resnet_glow(), *args, **kwargs):
        super().__init__(forward_min_event_ndims=3,
                         inverse_min_event_ndims=3,
                         *args, **kwargs)
        self.nn_ctor = nn_ctor
        self.nn = None
        self.forward_log_s = None
        self.inverse_log_s = None
        
    def _init_nn(self, x):
        if self.nn is None:
            self.nn = self.nn_ctor(x.shape[-1])
        
    def _forward(self, x):
        self._init_nn(x)
        x_a, x_b = tf.split(x, 2, axis=-1)
        log_s, t = self.nn.predict(x_b)
        self.forward_log_s = log_s
        s = tf.math.exp(log_s)
        y_a = s*x_a + t
        y_b = x_b
        return tf.concat([y_a, y_b], axis=-1)
    
    def _inverse(self, y):
        self._init_nn(y)
        y_a, y_b = tf.split(y, 2, axis=-1)
        log_s, t = self.nn.predict(y_b)
        self.inverse_log_s = log_s
        s = tf.math.exp(log_s)
        x_a = (y_a - t) / s
        x_b = y_b
        return tf.concat([x_a, x_b], axis=-1)
    
    def _forward_log_det_jacobian(self, x):
        if self.forward_log_s is None:
            self._forward(x)
        return tf.math.reduce_sum(self.forward_log_s)
    
    def _inverse_log_det_jacobian(self, y):
        if self.inverse_log_s is None:
            self._inverse(y)
        return tf.math.reduce_sum(self.inverse_log_s)