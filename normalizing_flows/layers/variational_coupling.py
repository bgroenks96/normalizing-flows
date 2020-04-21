import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp

def build_deterministic(input_shape, c, layer_fn, log_scale, name='deterministic', **kwargs):
    x = layers.Input(input_shape)
    s = layer_fn(c,f'{name}/s')(x)
    s = layers.Lambda(lambda x: x*tf.math.exp(log_scale))(s)
    s = layers.Activation(lambda x: tf.nn.sigmoid(x)+0.1)(s)
    t = layer_fn(c,f'{name}/t')(x)
    log_prob = layers.Lambda(lambda x: tf.zeros_like(x))(x)
    kld = layers.Lambda(lambda x: tf.zeros_like(x))(x)
    return tf.keras.Model(inputs=x, outputs=[s,t,log_prob,kld])

def build_gaussian_diag(input_shape, c, layer_fn, log_scale, prior, name='gaussian_diag', eps=1.0E-5, **kwargs):
    def _make_gaussian_diag(params):
        # use first half of params for mu
        mus = tf.gather(params, [i for i in range(c)], axis=-1)
        # second half for sigma
        sigmas = eps + tf.nn.softplus(tf.gather(params, [i for i in range(c,2*c)], axis=-1))
        return tfp.distributions.Normal(loc=mus, scale=sigmas)
    def _dist_tensor_fn(dist):
        y = dist.sample()
        log_prob = dist.log_prob(y)
        kld = tfp.distributions.kl_divergence(dist, prior)
        return y, log_prob, kld
    x = layers.Input(input_shape)
    s_params = layer_fn(2*c,f'{name}/s_vars')(x)
    t_params = layer_fn(2*c,f'{name}/t_vars')(x)
    s, s_lp, s_kld = tfp.layers.DistributionLambda(_make_gaussian_diag, _dist_tensor_fn)(s_params)
    t, t_lp, t_kld = tfp.layers.DistributionLambda(_make_gaussian_diag, _dist_tensor_fn)(t_params)
    s = layers.Activation(lambda x: tf.nn.sigmoid(x)+0.1)(s)
    log_prob = layers.Add()([s_lp, t_lp])
    kld = layers.Add()([s_kld, t_kld])
    return tf.keras.Model(inputs=x, outputs=[s,t,log_prob,kld])

class VariationalCoupling(layers.Layer):
    def __init__(self, c, layer_fn, dist_type='gaussian_diag', name='variational_coupling', dtype=tf.float32, **kwargs):
        """
        Initializes a new coupling layer which produces s and t for affine transformation sx + t,
        with s ~ q1 and t ~ q2, where q1,q2 are variational distributions. For deterministic output,
        a non-stochastic (point-mass) distribution is made available.
        
        c         : number of output dimensions
        layer_fn  : function: (c,name)->Layer which takes the number of dimensions and
                    produces a Layer with the appropriate dimensions.
        dist_type : type of distribution to use; defaults to 'gaussian_diag'
                    use 'deterministic' for non-stochastic optimization
        """
        super().__init__(self, name=name, dtype=dtype, **kwargs)
        self.num_dims = c
        self.layer_fn = layer_fn
        self.dist_type = dist_type
        self.log_scale = None
        self.coupling_fn = None
        
    def build(self, shape):
        self.log_scale = tf.Variable(tf.zeros((1,1,1,shape[-1]//2)), dtype=self.dtype, name=f'{self.name}/log_scale')
        c = self.num_dims
        if self.dist_type == 'deterministic':
            self.coupling_fn = build_deterministic(shape[1:], c, self.layer_fn, self.log_scale, name=self.name)
        elif self.dist_type == 'gaussian_diag':
            prior = tfp.distributions.Normal(loc=tf.zeros(shape[1:]), scale=0.5*tf.ones(shape[1:]))
            self.coupling_fn = build_gaussian_diag(shape[1:], c, self.layer_fn, self.log_scale, prior, name=self.name)
            
    def call(self, x):
        s, t, prior_log_prob, kld = self.coupling_fn(x)
        self.add_loss(kld)
        return s, t, prior_log_prob
