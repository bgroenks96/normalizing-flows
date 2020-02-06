import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections
import flows
from tqdm import tqdm

class FlowLVM(tf.Module):
    """
    Flow-based Latent Variable Model; attempts to learn a variational approximation
    for the joint distribution p(x,z) by minimizing the log likelihood of F^-1(x)
    under the prior p(z) where F is an invertible transformation z <--> x.
    """
    def __init__(self,
                 transform: flows.Transform,
                 prior: tfp.distributions.Distribution=tfp.distributions.Normal(loc=0.0, scale=1.0),
                 input_shape=None,
                 num_bins=None,
                 optimizer=tf.keras.optimizers.Adamax(lr=1.0E-3),
                 clip_grads=1.0,
                 name='flvm'):
        """
        transform     : a bijective transform to be applied to the initial variational density;
                        note that this is assumed to be a transform z -> x where the inverse is x -> z
        prior         : a tfp.distributions.Distribution representing the prior, p(z)
        input_shape   : the shape of the observed variables, x
        num_bins      : for discrete input spaces: number of discretized bins; i.e. num_bins = 2^(num_bits)
        optimizer     : optimizer to use during training
        clip_grads    : If not None and > 0, the gradient clipping ratio for clip_by_global_norm;
                        otherwise, no gradient clipping is applied
        """
        super().__init__(name=name)
        self.prior = prior
        self.transform = transform
        self.num_bins = num_bins
        self.optimizer = optimizer
        self.clip_grads = clip_grads
        self.scale_factor = np.log2(num_bins) if num_bins is not None else 1.0
        self.input_shape = input_shape
        if self.input_shape is not None:
            self.initialize(self.input_shape)
        
    def initialize(self, input_shape):
        self.input_shape = input_shape
        with tf.init_scope():
            self.transform.initialize(input_shape)
    
    @tf.function
    def _preprocess(self, x):
        if self.num_bins is not None:
            x += tf.random.uniform(x.shape, 0, 1./self.num_bins)
        return x
        
    @tf.function
    def eval_batch(self, x, **flow_kwargs):
        assert self.input_shape is not None, 'model not initialized'
        num_elements = tf.cast(x.shape[1]*x.shape[2]*x.shape[3], tf.float32)
        x = self._preprocess(x)
        z, ldj = self.transform.inverse(x, **flow_kwargs)
        prior_log_probs = self.prior.log_prob(z)
        if z.shape.rank > 1:
            # reduce log probs along non-batch dimensions
            prior_log_probs = tf.math.reduce_sum(prior_log_probs, axis=[i for i in range(1,z.shape.rank)])
        log_probs = prior_log_probs + ldj
        nll = -(log_probs - self.scale_factor*num_elements) / num_elements
        return tf.math.reduce_mean(nll), \
               -tf.math.reduce_mean(prior_log_probs) / num_elements, \
               tf.math.reduce_mean(ldj) / num_elements
        
    @tf.function
    def train_batch(self, x, **flow_kwargs):
        """
        Performs a single iteration of mini-batch SGD on input x.
        Returns loss, nll, prior, ildj[, grad_norm]
                where loss is the total optimized loss (including regularization),
                nll is the averaged negative log likelihood component,
                prior is the averaged prior negative log likelihodd,
                ildj is the inverse log det jacobian,
                and, if clip_grads is True, grad_norm is the global max gradient norm
        """
        assert self.input_shape is not None, 'model not initialized'
        nll, prior_nll, ldj = self.eval_batch(x, **flow_kwargs)
        reg_loss = self.transform._regularization_loss()
        objective = nll + reg_loss
        gradients = self.optimizer.get_gradients(objective, self.trainable_variables)
        if self.clip_grads:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_grads)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return objective, nll, prior_nll, ldj
            
    def train(self, train_data: tf.data.Dataset, steps_per_epoch, num_epochs=1, **flow_kwargs):
        train_data = train_data.take(steps_per_epoch).repeat(num_epochs)
        with tqdm(total=steps_per_epoch*num_epochs) as prog:
            hist = collections.deque(maxlen=steps_per_epoch)
            for epoch in range(num_epochs):
                for x in train_data.take(steps_per_epoch):
                    loss, nll, prior, _ = self.train_batch(x, **flow_kwargs)
                    hist.append((loss, nll, prior))
                    prog.update(1)
                    prog.set_postfix({'epoch': epoch+1,
                                      'loss': np.mean([record[0] for record in hist]),
                                      'nll': np.mean([record[1] for record in hist]),
                                      'prior': np.mean([record[2] for record in hist])})
                    
    def evaluate(self, validation_data: tf.data.Dataset, validation_steps, **flow_kwargs):
        validation_data = validation_data.take(validation_steps)
        with tqdm(total=validation_steps) as prog:
            hist = collections.deque(maxlen=validation_steps)
            for x in validation_data:
                nll, prior, _ = self.eval_batch(x, **flow_kwargs)
                hist.append((nll, prior))
                prog.update(1)
                prog.set_postfix({'nll': np.mean([record[0] for record in hist]),
                                  'prior': np.mean([record[1] for record in hist])})
                
    def encode(self, x):
        z, _ = self.transform.inverse(x)
        return z
    
    def decode(self, z):
        x, _ = self.transform.forward(z)
        return x
                
    def sample(self, n=1):
        assert self.input_shape is not None, 'model not initialized'
        event_ndims = self.prior.event_shape.rank
        z_shape = self.input_shape[1:]
        z = self.prior.sample((n,*z_shape[:len(z_shape)-event_ndims]))
        z = tf.reshape(z, (n, -1))
        return self.decode(z)
