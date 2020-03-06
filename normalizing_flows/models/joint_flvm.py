import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import normalizing_flows.utils as utils
from normalizing_flows.flows import Transform
from tqdm import tqdm
from .trackable_module import TrackableModule
from .losses import wasserstein_loss

class JointFlowLVM(TrackableModule):
    """
    Flow-based latent variable model for joint distribution inference.
    Given random variables X, Y; JointFlowLVM attempts to learn a latent variable
    model P(X,Y,Z) enabling conditional inference P(X|Y) and P(Y|X) via implicit
    integration over the shared latent variables Z. This is done by learning two
    bijective mappings X<->Z and Y<->Z via maximum likelihood in conjunction with
    adversarial losses on P(X) and P(Y). See AlignFlow (Grover et al. 2019) for details.
    """
    def __init__(self,
                 G_zx: Transform,
                 G_zy: Transform,
                 D_x: tf.keras.Model,
                 D_y: tf.keras.Model,
                 prior: tfp.distributions.Distribution=tfp.distributions.Normal(loc=0.0, scale=1.0),
                 input_shape=None,
                 num_bins=None,
                 Gx_aux_loss=lambda x,y: tf.constant(0.),
                 Gy_aux_loss=lambda x,y: tf.constant(0.),
                 adversarial_loss_ctor=wasserstein_loss,
                 optimizer_g=tf.keras.optimizers.Adam(lr=1.0E-4, beta_1=0.5, beta_2=0.9),
                 optimizer_dx=tf.keras.optimizers.Adam(lr=1.0E-4, beta_1=0.5, beta_2=0.9),
                 optimizer_dy=tf.keras.optimizers.Adam(lr=1.0E-4, beta_1=0.5, beta_2=0.9),
                 clip_grads=10.0,
                 name='joint_flvm'):
        assert G_zx.name != G_zy.name, 'generators must have unique names'
        super().__init__({'optimizer_g': optimizer_g, 'optimizer_dx': optimizer_dx, 'optimizer_dy': optimizer_dy}, name=name)
        self.G_zx = G_zx
        self.G_zy = G_zy
        self.D_x = D_x
        self.D_y = D_y
        self.prior = prior
        self.input_shape = input_shape
        self.num_bins = num_bins
        self.Gx_aux_loss = Gx_aux_loss
        self.Gy_aux_loss = Gy_aux_loss
        self.adv_loss_ctor = adversarial_loss_ctor
        self.optimizer_g = optimizer_g
        self.optimizer_dx = optimizer_dx
        self.optimizer_dy = optimizer_dy
        self.clip_grads = clip_grads
        self.scale_factor = np.log2(num_bins) if num_bins is not None else 0.0
        if self._is_initialized():
            self.initialize(self.input_shape)
            
    def _is_initialized(self):
        return self.input_shape is not None
            
    def initialize(self, input_shape):
        self.input_shape = input_shape
        self.Dx_loss, self.Gx_loss = self.adv_loss_ctor(self.D_x)
        self.Dy_loss, self.Gy_loss = self.adv_loss_ctor(self.D_y)
        with tf.init_scope():
            self.G_zx.initialize(input_shape)
            self.G_zy.initialize(input_shape)
        self._init_checkpoint()
    
    def _preprocess(self, x):
        if self.num_bins is not None:
            x += tf.random.uniform(x.shape, 0, 1./self.num_bins)
        return x
    
    def predict_y(self, x):
        z, _ = self.G_zx.inverse(x)
        y, _ = self.G_zy.forward(z)
        return y
    
    def predict_x(self, y):
        z, _ = self.G_zy.inverse(y)
        x, _ = self.G_zx.forward(z)
        return x
        
    @tf.function
    def eval_generators_on_batch(self, x, y):
        assert self.input_shape is not None, 'model not initialized'
        num_elements = tf.cast(x.shape[1]*x.shape[2]*x.shape[3], tf.float32)
        x = self._preprocess(x)
        y = self._preprocess(y)
        # compute generator outputs
        z_x, ildj_x = self.G_zx.inverse(x)
        y_x, _ = self.G_zy.forward(z_x)
        z_y, ildj_y = self.G_zy.inverse(y)
        x_y, _ = self.G_zx.forward(z_y)
        # compute adversarial losses
        gx_loss = self.Gx_loss(x, x_y)
        gy_loss = self.Gy_loss(y, y_x)
        # compute auxiliary loss
        gx_aux = self.Gx_aux_loss(y, x_y)
        gy_aux = self.Gy_aux_loss(x, y_x)
        # compute likelihood losses
        prior_logp_x = self.prior.log_prob(z_x)
        prior_logp_y = self.prior.log_prob(z_y)
        if prior_logp_x.shape.rank > 1:
            # reduce log probs along non-batch dimensions
            prior_logp_x = tf.math.reduce_sum(prior_logp_x, axis=[i for i in range(1,prior_logp_x.shape.rank)])
            prior_logp_y = tf.math.reduce_sum(prior_logp_y, axis=[i for i in range(1,prior_logp_y.shape.rank)])
        nll_x = -tf.math.reduce_mean((prior_logp_x + ildj_x - self.scale_factor*num_elements) / num_elements)
        nll_y = -tf.math.reduce_mean((prior_logp_y + ildj_y - self.scale_factor*num_elements) / num_elements)
        return nll_x, nll_y, gx_loss, gy_loss, gx_aux, gy_aux
    
    @tf.function
    def eval_discriminators_on_batch(self, x, y):
        x_pred = self.predict_x(y)
        y_pred = self.predict_y(x)
        # evaluate discriminators
        dx_loss = self.Dx_loss(x, x_pred)
        dy_loss = self.Dy_loss(y, y_pred)
        return dx_loss, dy_loss
        
    @tf.function
    def train_generators_on_batch(self, x, y, lam=1.0, alpha=1.0):
        assert self.input_shape is not None, 'model not initialized'
        nll_x, nll_y, gx_loss, gy_loss, gx_aux, gy_aux = self.eval_generators_on_batch(x, y)
        # compute losses
        reg_losses = [self.G_zx._regularization_loss()]
        reg_losses += [self.G_zy._regularization_loss()]
        g_obj = gx_loss + gy_loss + lam*(nll_x + nll_y) + alpha*(gx_aux + gy_aux) + tf.math.add_n(reg_losses)
        # generator gradient update
        generator_variables = list(self.G_zx.trainable_variables) + list(self.G_zy.trainable_variables)
        g_grads = tf.gradients(g_obj, generator_variables)
        if self.clip_grads:
            g_grads, grad_norm = tf.clip_by_global_norm(g_grads, self.clip_grads)
        self.optimizer_g.apply_gradients(zip(g_grads, generator_variables))
        return g_obj, nll_x, nll_y, gx_loss, gy_loss, gx_aux, gy_aux
    
    @tf.function
    def train_discriminators_on_batch(self, x, y):
        dx_loss, dy_loss = self.eval_discriminators_on_batch(x, y)
        dx_grads = tf.gradients(dx_loss, self.D_x.trainable_variables)
        dy_grads = tf.gradients(dy_loss, self.D_y.trainable_variables)
        self.optimizer_dx.apply_gradients(zip(dx_grads, self.D_x.trainable_variables))
        self.optimizer_dy.apply_gradients(zip(dy_grads, self.D_y.trainable_variables))
        return dx_loss, dy_loss
    
    def train(self, train_data: tf.data.Dataset, steps_per_epoch, num_epochs=1,
              lam=1.0, alpha=0.1, **flow_kwargs):
        train_gen_data = train_data.take(steps_per_epoch).repeat(num_epochs)
        with tqdm(total=steps_per_epoch*num_epochs, desc='train') as prog:
            hist = dict()
            for epoch in range(num_epochs):
                for x,y in train_gen_data.take(steps_per_epoch):
                    # train discriminators
                    dx_loss, dy_loss = self.train_discriminators_on_batch(x, y)
                    # train generators
                    g_obj, nll_x, nll_y,_,_,_,_ = self.train_generators_on_batch(x, y, lam=lam)
                    utils.update_metrics(hist, g_obj=g_obj.numpy(), dx_loss=dx_loss.numpy(), dy_loss=dy_loss.numpy(),
                                         nll_x=nll_x.numpy(), nll_y=nll_y.numpy())
                    prog.update(1)
                    prog.set_postfix(utils.get_metrics(hist))
        return hist
                    
    def evaluate(self, validation_data: tf.data.Dataset, validation_steps, **flow_kwargs):
        validation_data = validation_data.take(validation_steps)
        with tqdm(total=validation_steps, desc='eval') as prog:
            hist = dict()
            for x,y in validation_data:
                # train discriminators
                dx_loss, dy_loss = self.eval_discriminators_on_batch(x, y)
                # train generators
                nll_x, nll_y, gx_loss, gy_loss, gx_aux, gy_aux = self.eval_generators_on_batch(x, y)
                utils.update_metrics(hist,
                                     nll_x=nll_x.numpy(),
                                     nll_y=nll_y.numpy(),
                                     gx_loss=gx_loss.numpy(),
                                     gy_loss=gy_loss.numpy(),
                                     dx_loss=dx_loss.numpy(),
                                     dy_loss=dy_loss.numpy(),
                                     gx_aux=gx_aux.numpy(),
                                     gy_aux=gy_aux.numpy())
                prog.update(1)
                prog.set_postfix(utils.get_metrics(hist))
        return hist
                
    def encode_x(self, x):
        z, _ = self.G_zx.inverse(x)
        return z
    
    def decode_x(self, z):
        x, _ = self.G_zx.forward(z)
        return x
    
    def encode_y(self, y):
        y, _ = self.G_zy.inverse(y)
        return y
    
    def decode_y(self, z):
        y, _ = self.G_zy.forward(z)
        return y
                
    def sample(self, n=1):
        assert self.input_shape is not None, 'model not initialized'
        event_ndims = self.prior.event_shape.rank
        z_shape = self.input_shape[1:]
        if self.prior.is_scalar_batch():
            z = self.prior.sample((n,*z_shape[:len(z_shape)-event_ndims]))
        else:
            z = self.prior.sample((n,))
        return self.decode_x(z), self.decode_y(z)
    