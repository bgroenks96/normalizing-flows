import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections
import flows
from flows.glow import GlowFlow
from tqdm import tqdm
from tensorflow.keras import Model

class VariationalModel(tf.Module):
    def __init__(self,
                 prior: tf.Module,
                 parameterizer,
                 transform: flows.Transform=flows.Identity(),
                 output_shape=None,
                 num_bins=None,
                 optimizer=tf.keras.optimizers.Adamax(lr=1.0E-3),
                 clip_grads=True):
        """
        Creates a generalized, trainable model for variational inference.
        
        prior         : a callable tf.Module (or Keras Model) which represents an inference function
                        f: (X -> Theta) such that X is the model inputs and Theta is a tensor of
                        distribution parameters.
        parameterizer : a function f: (Theta -> tfp.distributions.Distribution) which parameterizes a
                        variational distribution from the parameter values produced by 'prior'
        transform     : a bijective transform to be applied to the initial variational density
        num_bins      : for discrete input spaces: number of discretized bins; i.e. num_bins = 2^(num_bits)
        optimizer     : optimizer to use during training
        clip_grads    : True if gradients should be clipped to the global norm, False otherwise
        """
        self.prior = prior
        self.parameterizer = parameterizer
        self.transform = transform
        self.num_bins = num_bins
        self.optimizer = optimizer
        self.clip_grads = clip_grads
        self.scale_factor = np.log2(num_bins) if num_bins is not None else 1.0
        self.output_shape = output_shape
        if self.output_shape is not None:
            self.initialize(self.output_shape)
        
    def initialize(self, output_shape):
        self.output_shape = output_shape
        with tf.init_scope():
            self.transform.initialize(output_shape)
        
    @tf.function
    def _prior_log_prob(self, params, z):
        prior_dist = self.parameterizer(params)
        return prior_dist.log_prob(z)
    
    @tf.function
    def _preprocess(self, x):
        if self.num_bins is not None:
            x += tf.random.uniform(x.shape, 0, 1./self.num_bins)
        return x
        
    @tf.function
    def eval_batch(self, x, y, **flow_kwargs):
        assert self.output_shape is not None, 'model not initialized'
        num_elements = tf.cast(y.shape[1]*y.shape[2]*y.shape[3], tf.float32)
        params = self.prior(x)
        y = self._preprocess(y)
        z, ldj = self.transform.inverse(y, **flow_kwargs)
        z = tf.reshape(z, tf.shape(y))
        prior_log_probs = tf.math.reduce_sum(self._prior_log_prob(params, z), axis=[1,2,3])
        log_probs = prior_log_probs + ldj
        nll = -(log_probs - self.scale_factor*num_elements) / num_elements
        return tf.math.reduce_mean(nll), \
               -tf.math.reduce_mean(prior_log_probs) / num_elements, \
               tf.math.reduce_mean(ldj) / num_elements
        
    @tf.function
    def train_batch(self, x, y, **flow_kwargs):
        """
        Performs a single iteration of mini-batch SGD on input x.
        Returns loss, nll, prior, ildj[, grad_norm]
                where loss is the total optimized loss (including regularization),
                nll is the averaged negative log likelihood component,
                prior is the averaged prior negative log likelihodd,
                ildj is the inverse log det jacobian,
                and, if clip_grads is True, grad_norm is the global max gradient norm
        """
        assert self.output_shape is not None, 'model not initialized'
        nll, prior_nll, ldj = self.eval_batch(x, y, **flow_kwargs)
        reg_losses = self.prior.get_losses_for(None) if isinstance(self.prior, Model) else []
        reg_losses += [self.transform._regularization_loss()]
        objective = nll + tf.math.add_n(reg_losses)
        gradients = self.optimizer.get_gradients(objective, self.trainable_variables)
        if self.clip_grads:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        num_elements = tf.cast(y.shape[1]*y.shape[2]*y.shape[3], tf.float32)
        return objective, nll, prior_nll, ldj
            
    def train(self, train_data: tf.data.Dataset, steps_per_epoch, num_epochs=1, supervised=False):
        train_data = train_data.take(steps_per_epoch).repeat(num_epochs)
        with tqdm(total=steps_per_epoch*num_epochs) as prog:
            hist = collections.deque(maxlen=steps_per_epoch)
            for epoch in range(num_epochs):
                for i, batch in enumerate(train_data.take(steps_per_epoch)):
                    params = batch if supervised else [tf.zeros(batch.shape), batch]
                    loss, nll, prior, _ = self.train_batch(*params)
                    hist.append((loss, nll, prior))
                    prog.update(1)
                    prog.set_postfix({'epoch': epoch+1,
                                      'loss': np.mean([record[0] for record in hist]),
                                      'nll': np.mean([record[1] for record in hist]),
                                      'prior': np.mean([record[2] for record in hist])})
                    
    def evaluate(self, validation_data: tf.data.Dataset, validation_steps, supervised=False):
        validation_data = validation_data.take(validation_steps)
        with tqdm(total=validation_steps) as prog:
            hist = collections.deque(maxlen=validation_steps)
            for batch in validation_data:
                params = batch if supervised else [tf.zeros(batch.shape), batch]
                nll, prior, _ = self.eval_batch(*params)
                hist.append((nll, prior))
                prog.update(1)
                prog.set_postfix({'nll': np.mean([record[0] for record in hist]),
                                  'prior': np.mean([record[1] for record in hist])})
                
    def predict_mean(self, x=None, n=1):
        x = tf.zeros((n, *self.output_shape[1:])) if x is None else x
        params = self.prior(x)
        prior = self.parameterizer(params)
        z = prior.mean()
        z = tf.reshape(z, (tf.shape(z)[0], -1))
        x, _ = self.transform.forward(z)
        return x
                
    def sample(self, x=None, n=1):
        x = tf.zeros((n, *self.output_shape[1:])) if x is None else x
        params = self.prior(x)
        prior = self.parameterizer(params)
        z = prior.sample()
        z = tf.reshape(z, (tf.shape(z)[0], -1))
        x, _ = self.transform.forward(z)
        return x
    
    def distribution(self, x=None, invert=False):
        x = tf.zeros((n, *self.output_shape[1:])) if x is None else x
        transform = flows.TransformBijector(self.transform)
        params = self.prior(x)
        prior = self.parameterizer(params)
        return transform(prior)
        
### Miscellaneous utils for standalone amortized VI ###

def nll_loss(distribution_fn):
    def nll(y_true, y_pred):
        def log_prob(dist: tfp.distributions.Distribution):
            return dist.log_prob(y_true)
        dist = tfp.layers.DistributionLambda(distribution_fn, log_prob)
        nll = -dist(y_pred)
        return tf.reduce_mean(nll, axis=-1)
    return nll

def parameterize(model: Model, distribution_fn):
    model.predict_mean = lambda x: distribution_fn(model.predict(x)).mean()
    model.predict_q = lambda x, q: distribution_fn(model.predict(x)).quantile(q)
    model.sample = lambda x: distribution_fn(model.predict(x)).sample()
    return model