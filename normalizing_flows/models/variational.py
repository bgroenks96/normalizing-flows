import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import normalizing_flows.flows as flows
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tqdm import tqdm
from typing import Callable

def nll_loss(distribution_fn, num_bins=None):
    """
    Implementation of normalized NLL loss function.
    
    distribution_fn : a callable that takes a tf.Tensor and returns a valid
                      TFP Distribution
    num_bins        : number of discretization bins; None for continuous data
    """
    scale_factor = np.log2(num_bins) if num_bins is not None else 1.0
    def nll(y_true, y_pred):
        def _preprocess(self, x):
            if num_bins is not None:
                x += tf.random.uniform(x.shape, 0, 1./num_bins)
            return x
        def log_prob(dist: tfp.distributions.Distribution):
            return dist.log_prob(y_true)
        num_elements = tf.math.reduce_prod(tf.cast(tf.shape(y_true)[1:], dtype=y_true.dtype))
        dist = tfp.layers.DistributionLambda(distribution_fn, log_prob)
        log_probs = tf.math.reduce_sum(dist(y_pred), axis=[i for i in range(1, y_true.shape.rank)])
        nll = -(log_probs - scale_factor*num_elements) / num_elements
        return nll
    return nll

class VariationalModel(Model):
    """
    Extension of Keras Model for supervised, variational inference networks.
    """
    def __init__(self,
                 distribution_fn: Callable[[tf.Tensor], tfp.distributions.Distribution],
                 transform: flows.Transform=flows.Identity(),
                 num_bins=None,
                 clip_grads=1.0,
                 *model_args,
                 **model_kwargs):
        """
        distribution_fn : a callable that takes a tf.Tensor and returns a valid
                          TFP Distribution
        transform       : a bijective transform to be applied to the initial density
                          returned by distribution_fn; defaults to Identity (no transform)
        num_bins        : number of discretization bins to use; defaults to None (continuous inputs)
        """
        super().__init__(*model_args, **model_kwargs)
        self.dist_fn = distribution_fn
        self.transform = transform
        self.num_bins = num_bins
        self.scale_factor = np.log2(num_bins) if num_bins is not None else 1.0
        self.clip_grads = clip_grads
        
    def compile(self, output_shape, optimizer, output_dtype=tf.float32, **kwargs):
        def nll(y_true, y_pred):
            return -self._log_prob(y_pred, y_true)
        assert 'loss' not in kwargs, 'NLL loss is automatically supplied by VariationalModel'
        self.optimizer = optimizer
        self.transform.initialize(output_shape)
        super().compile(loss=nll, target_tensors=K.placeholder(shape=output_shape, dtype=output_dtype), **kwargs)
        
    @tf.function
    def eval_batch(self, x, y, **flow_kwargs):
        assert self.input_shape is not None, 'model not initialized'
        return tf.math.reduce_mean(-self.log_prob(x, y))
        
    @tf.function
    def train_batch(self, x, y, **flow_kwargs):
        """
        Performs a single iteration of mini-batch SGD on input x, y.
        Returns loss, nll, prior, ildj[, grad_norm]
                where loss is the total optimized loss (including regularization),
                nll is the averaged negative log likelihood component,
                prior is the averaged prior negative log likelihodd,
                ildj is the inverse log det jacobian,
                and, if clip_grads is True, grad_norm is the global max gradient norm
        """
        assert self.input_shape is not None, 'model not initialized (did you call compile?)'
        assert self.optimizer is not None, 'model not initialized (did you call compile?)'
        nll = self.eval_batch(x, y, **flow_kwargs)
        reg_losses = self.get_losses_for(None)
        reg_losses += [self.transform._regularization_loss()]
        objective = nll + tf.math.add_n(reg_losses)
        gradients = self.optimizer.get_gradients(objective, self.trainable_variables)
        if self.clip_grads:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_grads)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return objective, nll
        
    def fit(self, data: tf.data.Dataset, epochs=1, steps_per_epoch=1,
            validation_data=None, validation_steps=1,
            **flow_kwargs):
        data.repeat(epochs)
        if validation_data is not None:
            validation_data = validation_data.repeat(epochs)
        for epoch in range(epochs):
            with tqdm(total=steps_per_epoch, desc=f'train, epoch {epoch+1}/{epochs}') as prog:
                avg_nll = 0.0
                avg_loss = 0.0
                for i, (x, y) in enumerate(data.take(steps_per_epoch)):
                    loss, nll = self.train_batch(x, y, **flow_kwargs)
                    avg_nll = (nll.numpy() + i*avg_nll) / (i+1)
                    avg_loss = (loss.numpy() + i*avg_loss) / (i+1)
                    prog.update(1)
                    prog.set_postfix({'loss': avg_loss, 'nll': avg_nll})
            with tqdm(total=steps_per_epoch, desc=f'test, epoch {epoch+1}/{epochs}') as prog:
                if validation_data is None:
                    continue
                avg_nll_test = 0.0
                for i, (x, y) in enumerate(validation_data.take(validation_steps)):
                    nll = self.eval_batch(x, y, **flow_kwargs)
                    avg_nll_test = (nll.numpy() + i*avg_nll_test) / (i+1)
                    prog.update(1)
                    prog.set_postfix({'nll': avg_nll_test})
        return self
        
    def log_prob(self, x, y):
        params = self(x)
        return self._log_prob(params, y)
        
    def predict_mean(self, x):
        assert self.transform.has_constant_ldj, 'mean not defined for transforms with variable logdetJ'
        params = self(x)
        dist = self.dist_fn(params)
        z = dist.mean()
        y, _ = self.transform.forward(tf.reshape(z, (tf.shape(z)[0], -1)))
        return y
    
    def sample(self, x, sample_fn=None):
        params = self(x)
        dist = self.dist_fn(params)
        if sample_fn is not None:
            z = sample_fn(dist)
        else:
            z = dist.sample()
        z = (z - params[:,:,:,:1]) / (1.0E-6 + tf.math.exp(params[:,:,:,1:]))
        y, _ = self.transform.forward(tf.reshape(z, (tf.shape(z)[0], -1)))
        return y
        
    def quantile(self, x, q):
        params = self(x)
        dist = self.dist_fn(params)
        z = dist.quantile()
        y, _ = self.transform.forward(tf.reshape(z, (tf.shape(z)[0], -1)))
        return y
    
    def _log_prob(self, params, y):
        if self.num_bins is not None:
            y += tf.random.uniform(y.shape, 0, 1./self.num_bins)
        num_elements = tf.math.reduce_prod(tf.cast(tf.shape(y)[1:], dtype=y.dtype))
        dist = self.dist_fn(params)
        z, ildj = self.transform.inverse(y)
        z = tf.reshape(z, tf.shape(y))
        prior_log_probs = dist.log_prob(z*(1.0E-6+tf.math.exp(params[:,:,:,1:])) + params[:,:,:,:1])
        prior_log_probs = tf.math.reduce_sum(prior_log_probs, axis=[i for i in range(1, prior_log_probs.shape.rank)])
        log_probs = (prior_log_probs + ildj - self.scale_factor*num_elements) / num_elements
        return log_probs