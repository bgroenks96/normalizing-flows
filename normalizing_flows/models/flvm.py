import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import normalizing_flows.flows as flows
import normalizing_flows.utils as utils
from tqdm import tqdm
from .trackable_module import TrackableModule

class FlowLVM(TrackableModule):
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
                 cond_fn=None,
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
        super().__init__({'optimizer': optimizer}, name=name)
        self.prior = prior
        self.transform = transform
        self.num_bins = num_bins
        self.cond_fn = cond_fn
        self.optimizer = optimizer
        self.clip_grads = clip_grads
        self.scale_factor = np.log2(num_bins) if num_bins is not None else 0.0
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
    def eval_cond(self, z, y):
        y_pred = self.cond_fn(z)
        return tf.math.reduce_mean((y - y_pred)**2)
        
    @tf.function
    def eval_batch(self, x, **flow_kwargs):
        assert self.input_shape is not None, 'model not initialized'
        num_elements = tf.cast(x.shape[1]*x.shape[2]*x.shape[3], tf.float32)
        x = self._preprocess(x)
        z, ldj = self.transform.inverse(x, **flow_kwargs)
        y_loss = 0.0
        if self.cond_fn is not None and 'y_cond' in flow_kwargs:
            y_loss = self.eval_cond(z, flow_kwargs['y_cond'])
        prior_log_probs = self.prior.log_prob(z)
        if prior_log_probs.shape.rank > 1:
            # reduce log probs along non-batch dimensions
            prior_log_probs = tf.math.reduce_sum(prior_log_probs, axis=[i for i in range(1,z.shape.rank)])
        log_probs = prior_log_probs + ldj
        nll = -(log_probs - self.scale_factor*num_elements) / num_elements
        return tf.math.reduce_mean(nll), \
               tf.math.reduce_mean(ldj / num_elements), \
               y_loss
        
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
        nll, ldj, y_loss = self.eval_batch(x, **flow_kwargs)
        reg_loss = self.transform._regularization_loss()
        objective = nll + reg_loss + y_loss
        gradients = tf.gradients(objective, self.trainable_variables)
        if self.clip_grads:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_grads)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return objective, nll, ldj
            
    def train(self, train_data: tf.data.Dataset, steps_per_epoch, num_epochs=1, conditional=False, **flow_kwargs):
        train_data = train_data.take(steps_per_epoch).repeat(num_epochs)
        with tqdm(total=steps_per_epoch*num_epochs) as prog:
            hist = dict()
            init = tf.constant(True) # init variable for data-dependent initialization
            for epoch in range(num_epochs):
                for batch in train_data.take(steps_per_epoch):
                    if conditional:
                        x, y = batch
                        loss, nll, ldj  = self.train_batch(x, y_cond=y, init=init, **flow_kwargs)
                    else:
                        x = batch
                        loss, nll, ldj  = self.train_batch(x, init=init, **flow_kwargs)
                    init=tf.constant(False)
                    utils.update_metrics(hist, loss=loss.numpy(), nll=nll.numpy())
                    prog.update(1)
                    prog.set_postfix({k: v[0] for k,v in hist.items()})
                    
    def evaluate(self, validation_data: tf.data.Dataset, validation_steps, conditional=False, **flow_kwargs):
        validation_data = validation_data.take(validation_steps)
        with tqdm(total=validation_steps) as prog:
            hist = dict()
            for batch in validation_data:
                if conditional:
                    x, y = batch
                    nll, ldj, y_loss  = self.eval_batch(x, y_cond=y, **flow_kwargs)
                else:
                    x = batch
                    nll, ldj, y_loss  = self.eval_batch(x, **flow_kwargs)
                utils.update_metrics(hist, nll=nll.numpy())
                prog.update(1)
                prog.set_postfix({k: v[0] for k,v in hist.items()})
                
    def encode(self, x, y_cond=None):
        if y_cond is not None:
            z, _ = self.transform.inverse(x, y_cond=y_cond)
        else:
            z, _ = self.transform.inverse(x)
        return z
    
    def decode(self, z, y_cond=None):
        if y_cond is not None:
            x, _ = self.transform.forward(z, y_cond=y_cond)
        else:
            x, _ = self.transform.forward(z)
        return x
                
    def sample(self, n=1, y_cond=None):
        assert self.input_shape is not None, 'model not initialized'
        batch_size = 1 if y_cond is None else y_cond.shape[0]
        event_ndims = self.prior.event_shape.rank
        z_shape = self.input_shape[1:]
        z = self.prior.sample((n*batch_size,*z_shape[:len(z_shape)-event_ndims]))
        z = tf.reshape(z, (n*batch_size, -1))
        if y_cond is not None:
            # repeat y_cond n times along batch axis
            y_cond = tf.keras.backend.repeat_elements(y_cond, n, axis=0)
        return self.decode(z, y_cond=y_cond)