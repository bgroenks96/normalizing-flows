import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Flatten, Reshape, Conv2D
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import Adamax
from normalizing_flows.flows import Flow
from normalizing_flows.layers import GatedConv2D, GatedConv2DTranspose, FlowLayer

class GatedConvVAE(tf.Module):
    """
    Gated, convolutional variational autoencoder with support for normalizing flows.
    """
    def __init__(self, img_wt, img_ht, flow: Flow = None, hidden_units=32, z_size=64,
                 encoder_strides=[2,2], decoder_strides=[2,2],
                 callbacks=[], metrics=[], output_activation='sigmoid', loss='binary_crossentropy',
                 beta_update_fn=None):
        super(GatedConvVAE, self).__init__()
        if beta_update_fn is None:
            beta_update_fn = lambda i, beta: 1.0E-2*i
        self.flow = flow
        self.hidden_units = hidden_units
        self.z_size = z_size
        self.num_downsamples = len(encoder_strides)
        self.num_upsamples = len(decoder_strides)
        self.encoder_strides = encoder_strides
        self.decoder_strides = decoder_strides
        self.output_activation = output_activation
        self.encoder = self._create_encoder(img_wt, img_ht)
        self.decoder, self.flow_layer = self._create_decoder(img_wt, img_ht)
        beta_update = LambdaCallback(on_epoch_begin=lambda i,_: beta_update_fn(i, self.flow_layer.beta))
        self.callbacks = [beta_update]+callbacks
        decoder_output = self.decoder(self.encoder(self.encoder.inputs))
        self.model = Model(inputs=self.encoder.inputs, outputs=decoder_output[0])
        self.model.compile(loss=loss, optimizer=Adamax(learning_rate=1.0E-4, clipnorm=1.), metrics=metrics)

    def fit(self, *args, **kwargs):
        """
        Passthrough to tf.keras.Model::fit
        """
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Passthrough to tf.keras.Model::predict
        """
        return self.model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """
        Passthrough to tf.keras.Model::evaluate
        """
        return self.model.evaluate(*args, **kwargs)
    
    def sample(self, x, n=1):
        """
        Sample from the conditional distribution Z ~ P(z|x)
        
        Returns  a tuple (x', zs) where x' is the reconstructed input and
        zs = [z_0, z_1, ... , z_k] where k is the number of flows.
        """
        input_shape = tf.shape(x)
        # add sample dim
        x = tf.expand_dims(x, axis=1)
        # broadcast according to number of samples
        x = tf.broadcast_to(x, (input_shape[0], n, *input_shape[1:]))
        # fold sample dim back into batch axis
        x = tf.reshape(x, (input_shape[0]*n, *input_shape[1:]))
        # encode/decode inputs and retrieve samples
        if self.flow is not None:
            z_mu, z_log_sigma, params = self.encoder.predict(x)
            outputs = self.decoder.predict([z_mu, z_log_sigma, params])
        else:
            z_mu, z_log_sigma = self.encoder.predict(x)
            outputs = self.decoder.predict([z_mu, z_log_sigma])
        # return x', (z_0, ..., z_k)
        return outputs[0], outputs[1:]

    def _conv_downsample(self, f, strides, x):
        g = GatedConv2D(f, 3, activation='linear')
        g_downsample = GatedConv2D(f, 3, strides=strides)
        return g_downsample(g(x))

    def _conv_upsample(self, f, strides, x):
        g = GatedConv2DTranspose(f, 3, activation='linear')
        g_upsample = GatedConv2DTranspose(f, 3, strides=strides)
        return g_upsample(g(x))

    def _create_encoder(self, wt, ht):
        input_0 = Input((wt, ht, 1))
        h = input_0
        for i in range(self.num_downsamples):
            h = self._conv_downsample(self.hidden_units*(i+1), self.encoder_strides[i], h)
        z_mu = Dense(self.z_size, activation='linear')(Flatten()(h))
        z_log_var = Dense(self.z_size, activation='linear')(Flatten()(h))
        outputs = [z_mu, z_log_var]
        if self.flow is not None:
            z_shape = tf.TensorShape((None,self.z_size))
            params = Dense(self.flow.param_count(z_shape), activation='linear')(Flatten()(h))
            outputs += [params]
        return Model(inputs=input_0, outputs=outputs)


    def _create_decoder(self, wt, ht):
        z_mu = Input(shape=(self.z_size,))
        z_log_var = Input(shape=(self.z_size,))
        inputs = [z_mu, z_log_var]
        if self.flow is not None:
            z_shape = tf.TensorShape((None, self.z_size))
            params = Input(shape=(self.flow.param_count(z_shape),))
            inputs += [params]
            self.flow.initialize(z_shape)
        flow_layer = FlowLayer(self.flow, min_beta=1.0E-3)
        zs, ldj, kld = flow_layer(inputs)
        z_k = zs[-1]
        s = np.prod(self.encoder_strides)
        h_k = Dense(wt*ht // s**2, activation='linear')(z_k)
        h_k = Reshape((wt // s, ht // s, 1))(h_k)
        for i in range(self.num_upsamples):
            h_k = self._conv_upsample(self.hidden_units*(i+1), self.decoder_strides[i], h_k)
        output_0 = Conv2D(1, 1, activation=self.output_activation, padding='same')(h_k)
        return Model(inputs=inputs, outputs=[output_0] + zs), flow_layer
