import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections
from flows import Transform
from models import adversarial, utils
from tqdm import tqdm

class JointFlowLVM(tf.Module):
    def __init__(self,
                 G_zx: Transform,
                 G_zy: Transform,
                 D_x: tf.keras.Model,
                 D_y: tf.keras.Model,
                 prior: tfp.distributions.Distribution=tfp.distributions.Normal(loc=0.0, scale=1.0),
                 input_shape=None,
                 num_bins=None,
                 optimizer_g=tf.keras.optimizers.Adam(lr=1.0E-4),
                 optimizer_dx=tf.keras.optimizers.Adam(lr=1.0E-4),
                 optimizer_dy=tf.keras.optimizers.Adam(lr=1.0E-4),
                 clip_grads=1.0,
                 name='flvm'):
        super().__init__(name=name)
        self.G_zx = G_zx
        self.G_zy = G_zy
        self.D_x = D_x
        self.D_y = D_y
        self.prior = prior
        self.input_shape = input_shape
        self.num_bins = num_bins
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
        with tf.init_scope():
            self.G_zx.initialize(input_shape)
            self.G_zy.initialize(input_shape)
            
    def _preprocess(self, x):
        if self.num_bins is not None:
            x += tf.random.uniform(x.shape, 0, 1./self.num_bins)
        return x
    
    def _eval_likelihood(self, transform, x):
        z, ldj = transform.inverse(x)
        prior_log_probs = self.prior.log_prob(z)
        if prior_log_probs.shape.rank > 1:
            # reduce log probs along non-batch dimensions
            prior_log_probs = tf.math.reduce_sum(prior_log_probs, axis=[i for i in range(1,z.shape.rank)])
        return prior_log_probs + ldj
        
    @tf.function
    def eval_batch(self, x, y, **flow_kwargs):
        assert self.input_shape is not None, 'model not initialized'
        num_elements = tf.cast(x.shape[1]*x.shape[2]*x.shape[3], tf.float32)
        x = self._preprocess(x)
        y = self._preprocess(y)
        # compute generator predictions
        z_x, ildj_x = self.G_zx.inverse(x)
        y_x, _ = self.G_zy.forward(z_x)
        z_y, ildj_y = self.G_zy.inverse(y)
        x_y, _ = self.G_zx.forward(z_y)
        # compute discriminator predictions
        dx_pred_real = self.D_x(x)
        dx_pred_gen = self.D_x(x_y)
        dy_pred_real = self.D_y(y)
        dy_pred_gen = self.D_y(y_x)
        # compute discriminator losses
        dx_loss = adversarial.loss(dx_pred_real, dx_pred_gen)
        dy_loss = adversarial.loss(dy_pred_real, dy_pred_gen)
        # compute generator adversarial losses (flip labels)
        gx_loss = adversarial.loss(dx_pred_gen, dx_pred_real)
        gy_loss = adversarial.loss(dy_pred_gen, dy_pred_real)
        # compute likelihood losses
        prior_logp_x = self.prior.log_prob(z_x)
        prior_logp_y = self.prior.log_prob(z_y)
        if prior_logp_x.shape.rank > 1:
            # reduce log probs along non-batch dimensions
            prior_logp_x = tf.math.reduce_sum(prior_logp_x, axis=[i for i in range(1,prior_logp_x.shape.rank)])
            prior_logp_y = tf.math.reduce_sum(prior_logp_y, axis=[i for i in range(1,prior_logp_y.shape.rank)])
        nll_x = -(prior_logp_x + ildj_x - self.scale_factor*num_elements) / num_elements
        nll_y = -(prior_logp_y + ildj_y - self.scale_factor*num_elements) / num_elements
        return nll_x, nll_y, gx_loss, gy_loss, dx_loss, dy_loss
        
    @tf.function
    def train_batch(self, x, y, lam=1.0, **flow_kwargs):
        assert self.input_shape is not None, 'model not initialized'
        nll_x, nll_y, gx_loss, gy_loss, dx_loss, dy_loss = self.eval_batch(x, y, **flow_kwargs)
        # compute losses
        reg_losses = [self.G_zx._regularization_loss()]
        reg_losses += [self.G_zy._regularization_loss()]
        g_obj = gx_loss + gy_loss + lam*nll_x + lam*nll_y + tf.math.add_n(reg_losses)
        g_obj = tf.math.reduce_mean(g_obj)
        # discriminator gradient update
        dx_loss = tf.math.reduce_mean(dx_loss)
        dy_loss = tf.math.reduce_mean(dy_loss)
        dx_grads = tf.gradients(dx_loss, self.D_x.trainable_variables)
        dy_grads = tf.gradients(dy_loss, self.D_y.trainable_variables)
        self.optimizer_dx.apply_gradients(zip(dx_grads, self.D_x.trainable_variables))
        self.optimizer_dy.apply_gradients(zip(dy_grads, self.D_y.trainable_variables))
        # generator gradient update
        generator_variables = list(self.G_zx.trainable_variables) + list(self.G_zy.trainable_variables)
        g_grads = tf.gradients(g_obj, generator_variables)
        if self.clip_grads:
            g_grads, grad_norm = tf.clip_by_global_norm(g_grads, self.clip_grads)
        self.optimizer_g.apply_gradients(zip(g_grads, generator_variables))
        nll_x = tf.math.reduce_mean(nll_x)
        nll_y = tf.math.reduce_mean(nll_y)
        return g_obj, dx_loss, dy_loss, nll_x, nll_y
    
    def train(self, train_data: tf.data.Dataset, steps_per_epoch, num_epochs=1, **flow_kwargs):
        train_data = train_data.take(steps_per_epoch).repeat(num_epochs)
        with tqdm(total=steps_per_epoch*num_epochs) as prog:
            hist = dict()
            init = tf.constant(True) # init variable for data-dependent initialization
            for epoch in range(num_epochs):
                for x,y in train_data.take(steps_per_epoch):
                    g_obj, dx_loss, dy_loss, nll_x, nll_y  = self.train_batch(x, y, **flow_kwargs)
                    init=tf.constant(False)
                    utils.update_metrics(hist, g_obj=g_obj.numpy(), dx_loss=dx_loss.numpy(), dy_loss=dy_loss.numpy(),
                                         nll_x=nll_x.numpy(), nll_y=nll_y.numpy())
                    prog.update(1)
                    prog.set_postfix({k: v[0] for k,v in hist.items()})
                    
    def evaluate(self, validation_data: tf.data.Dataset, validation_steps, **flow_kwargs):
        validation_data = validation_data.take(validation_steps)
        with tqdm(total=validation_steps) as prog:
            hist = dict()
            for x,y in validation_data:
                nll_x, nll_y, gx_loss, gy_loss, dx_loss, dy_loss  = self.eval_batch(x, y, **flow_kwargs)
                utils.update_metrics(hist,
                                     nll_x=tf.math.reduce_mean(nll_x).numpy(),
                                     nll_y=tf.math.reduce_mean(nll_y).numpy(),
                                     gx_loss=tf.math.reduce_mean(gx_loss).numpy(),
                                     gy_loss=tf.math.reduce_mean(gy_loss).numpy(),
                                     dx_loss=tf.math.reduce_mean(dx_loss).numpy(),
                                     dy_loss=tf.math.reduce_mean(dy_loss).numpy())
                prog.update(1)
                prog.set_postfix({k: v[0] for k,v in hist.items()})
                
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
                
    def sample(self, n=1, y_cond=None):
        assert self.input_shape is not None, 'model not initialized'
        batch_size = 1 if y_cond is None else y_cond.shape[0]
        event_ndims = self.prior.event_shape.rank
        z_shape = self.input_shape[1:]
        z = self.prior.sample((n*batch_size,*z_shape[:len(z_shape)-event_ndims]))
        z = tf.reshape(z, (n*batch_size, -1))
        return self.decode_x(z), self.decode_y(z)
    