"""
Generic interface for variational reparameterization.
"""
import tensorflow as tf
import tensorflow_probability as tfp
from normalizing_flows.utils import var
tfd = tfp.distributions

def reparameterize(initial_value, dist_type='gaussian_diag', name='reparamterized_variable', **kwargs):
    """
    Creates a new reparameterized variable w/ given initial value and distribution type.
    If dist_type = 'deterministic', the reparameterized variable is effectively equivalent
    to a standard, deterministic TF variable.
    """
    if dist_type == 'deterministic':
        return DeterministicVariable(initial_value, name=name)
    elif dist_type == 'gaussian_diag':
        return GaussianDiagVariable(initial_value, name=name, **kwargs)
    else:
        raise Exception(f'Unknown reparameterization distribution type: {dist_type}')
    
class ReparameterizedVariable(tf.Module):
    def __init__(self, name):
        super().__init__(name=name)
        
    def sample(self):
        x = self._sample()
        log_prob = self.prior.log_prob(x)
        return x, log_prob
    
    def regularization_loss(self):
        return self._kl_divergence()

class DeterministicVariable(ReparameterizedVariable):
    def __init__(self, value, name='deterministic_variable', alpha=1.0E-4):
        super().__init__(name)
        self.alpha = alpha
        self.var = tf.Variable(value, trainable=True, name=self.name)
        
    def _sample(self):
        return self.var
    
    def _regularization_loss(self):
        # L2 regularization
        return alpha*tf.math.reduce_sum(self.var**2)

class GaussianDiagVariable(ReparameterizedVariable):
    def __init__(self, mean, stddev=0.5, name='gaussian_variable'):
        super().__init__(name)
        self.prior = tfd.Normal(loc=mean, scale=stddev*tf.ones_like(mean))
        self.mean = tf.Variable(tf.random.normal(mean.shape, mean=self.prior.loc, stddev=0.1), trainable=True, name=f'{self.name}/mu')
        self.std = tf.Variable(tf.random.normal(mean.shape, mean=tfp.math.softplus_inverse(self.prior.scale), stddev=0.1), 
                               constraint=lambda s: 1.0E-5+tf.nn.softplus(s), trainable=True, name=f'{self.name}/std')
        self.variational_posterior = tfd.Normal(loc=self.mean, scale=self.std)
        
    def _sample(self):
        x = self.variational_posterior.sample()
        return x
    
    def _regularization_loss(self):
        # KL-divergence w/ prior
        return tfp.distributions.kl_divergence(self.variational_posterior, self.prior)