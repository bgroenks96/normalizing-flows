import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections
from flows import TransformBijector, Invert
from flows.glow import GlowFlow
from tqdm import tqdm

class Glow(tf.Module):
    def __init__(self,
                 prior,
                 prior_parameterize=None,
                 num_layers=1, depth_per_layer=1,
                 num_bins=None,
                 optimizer=tf.keras.optimizers.Adamax(lr=1.0E-3),
                 clip_grads=True,
                 *glow_args, **glow_kwargs):
        """
        Creates a new Glow model for variational inference.
        
        prior : either a tfp.distributions.Distribution instance that represents the variational prior,
                or a Keras Model f: (X -> Theta) where X is the model inputs and Theta is a tensor of distribution parameters.
                In the latter case, prior_parameterize must also be provided.
        prior_parameterize : a function f: (Theta -> tfp.distributions.Distribution) which parameterizes a distribution
                             from the parameter values produced by 'prior'
        num_layers : number of "layers" in the Glow flow; see Kingma and Dhariwal 2018
        depth_per_layer : number of Glow steps per layer; see Kingma and Dhariwal 2018
        optimizer : optimizer to use during training
        clip_grads : True if gradients should be clipped to the global norm, False otherwise
        glow_args : additional arguments for GlowFlow
        glow_kwargs : additional keyword arguments for GlowFlow
        """
        if not isinstance(prior, tfp.distributions.Distribution):
            assert isinstance(prior, tf.keras.Model), 'callable prior should be of type tf.keras.Model'
            assert prior_parameterize is not None, 'prior_parameterize must be provided for conditional priors'
        self.glow = GlowFlow(num_layers=num_layers, depth=depth_per_layer, *glow_args, **glow_kwargs)
        self.prior = prior
        self.prior_parameterize = prior_parameterize
        self.num_bins = num_bins
        self.optimizer = optimizer
        self.clip_grads = clip_grads
        self.prior_shape = None
        
    def _prior(self, x):
        if isinstance(self.prior, tfp.distributions.Distribution):
            prior = self.prior
            reg_losses = []
        else:
            prior_params = self.prior(x)
            prior = self.prior_parameterize(prior_params)
            reg_losses = self.prior.get_losses_for(None)
        if self.prior_shape is None:
            self.prior_shape = prior.batch_shape + prior.event_shape
            with tf.init_scope():
                self.glow.initialize(self.prior_shape)
        return prior, reg_losses
    
    def _preprocess(self, x):
        if self.num_bins is not None:
            x += tf.random.uniform(x.shape, 0, 1./self.num_bins)
        return x
    
    def initialize(self, input_shape):
        self.glow.initialize(input_shape)
        
    def train_batch(self, x, y=None):
        """
        Performs a single iteration of mini-batch SGD on input x.
        Returns loss, nll, prior, ildj[, grad_norm]
                where loss is the total optimized loss (including regularization),
                nll is the averaged negative log likelihood component,
                prior is the averaged prior negative log likelihodd,
                ildj is the inverse log det jacobian,
                and, if clip_grads is True, grad_norm is the global max gradient norm
        """
        with tf.GradientTape() as tape:
            y = x if y is None else y
            num_elements = tf.cast(x.shape[1]*x.shape[2]*x.shape[3], tf.float32)
            prior, reg_losses = self._prior(x)
            z, ldj = self.glow.forward(y)
            prior_log_probs = tf.math.reduce_sum(prior.log_prob(z), axis=[1,2,3])
            reg_losses += [self.glow._regularization_loss()]
            log_probs = prior_log_probs + ldj
            nll_loss = -tf.math.reduce_mean(log_probs)
            if self.num_bins is not None:
                nll_loss = (nll_loss + np.log(self.num_bins)*num_elements) / (np.log(2)*num_elements)
            total_loss = nll_loss + tf.math.add_n(reg_losses)
            gradients = tape.gradient(total_loss, self.trainable_variables)
            if self.clip_grads:
                gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss, nll_loss, -tf.math.reduce_mean(prior_log_probs / num_elements), tf.math.reduce_mean(ldj)
            
    def train(self, dataset: tf.data.Dataset, steps_per_epoch, num_epochs=1, has_y=False):
        train_dataset = dataset.take(steps_per_epoch).repeat(num_epochs)
        with tqdm(total=steps_per_epoch*num_epochs) as prog:
            hist = collections.deque(maxlen=steps_per_epoch)
            i = 0
            for batch in train_dataset:
                i += 1
                if has_y:
                    x, y = batch
                    loss, nll, prior, ldj = self.train_batch(x, y)
                else:
                    x = batch
                    loss, nll, prior, ldj = self.train_batch(x)
                hist.append((loss, nll, prior, ldj))
                prog.update(1)
                prog.set_postfix({'epoch': (i // steps_per_epoch) + 1,
                                  'loss': np.mean([record[0] for record in hist]),
                                  'nll': np.mean([record[1] for record in hist]),
                                  'prior': np.mean([record[2] for record in hist]),
                                  'ildj': np.mean([record[3] for record in hist])})
                
    def predict_mean(self, x=None):
        assert self.glow.input_shape is not None, 'model not initialized'
        if isinstance(self.prior, tfp.distributions.Distribution):
            z = self.prior.mean()
        else:
            assert x is not None, 'x must be provided for conditional prior'
            params = self.prior(x)
            prior = self.prior_parameterize(params)
            z = prior.mean()
        return self.glow.inverse(z)
    
    def distribution(self, x=None):
        assert self.glow.input_shape is not None, 'model not initialized'
        # Glow defines a forward pass as x -> z, whereas TFP bijectors
        # define it as z -> x; so we invert the glow transform prior to
        # constructing the TFP bijector
        transform = TransformBijector(Invert(self.glow))
        if isinstance(self.prior, tfp.distributions.Distribution):
            return transform(self.prior)
        else:
            assert x is not None, 'x must be provided for conditional prior'
            params = self.prior(x)
            prior = self.prior_parameterize(params)
            return transform(prior)
                
    def sample(self, x=None):
        return self.distribution(x).sample()
    