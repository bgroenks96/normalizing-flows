import tensorflow as tf
import tensorflow.keras.layers as layers

class FlowLayer(layers.Layer):
    def __init__(self, flow, min_beta=0.01, max_beta=1.0):
        super(FlowLayer, self).__init__()
        self.flow = flow
        self.min_beta = tf.constant(min_beta)
        self.max_beta = tf.constant(max_beta)
        self.beta = tf.Variable(min_beta, dtype=tf.float32, trainable=False)

    def call(self, inputs):
        """
        Requires three parameters: z_mu, z_sigma, params
        Returns reparameterized z_0, transformed z_k, summed log det jacobian, and KL divergence
        """
        assert len(inputs) == 3, 'expected three parameters, got {}'.format(len(inputs))
        z_mu, z_log_var, params = inputs
        # reparameterize z_mu, z_log_var
        z_var = tf.exp(z_log_var)
        z_0 = self.reparameterize(z_mu, z_var)
        # compute forward flow
        z_k, ldj = self.flow.forward(z_0, params)
        # compute KL divergence loss
        log_qz0 = tf.reduce_sum(-0.5*(z_log_var + (z_0 - z_mu)**2 / z_var), axis=1)
        log_pzk = tf.reduce_sum(-0.5*z_k**2, axis=1)
        kld = tf.reduce_mean(log_qz0 - log_pzk - ldj)
        self.add_loss(self.beta*kld)
        return z_0, z_k, ldj, kld

    def reparameterize(self, mu, var):
        eps = tf.random.normal(shape=tf.shape(mu))
        sigma = tf.sqrt(var)
        return mu + sigma*eps

    def set_beta(value):
        self.beta.assign(tf.maximum(tf.minimum(value, self.max_beta), self.min_beta))
