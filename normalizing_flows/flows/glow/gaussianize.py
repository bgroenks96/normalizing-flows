import tensorflow as tf
import tensorflow_probability as tfp
from normalizing_flows.flows import Transform
from . import Parameterize

def gaussianize(x, mus, log_sigmas, inverse=tf.constant(False)):
    if inverse:
        z = tf.math.exp(log_sigmas)*x + mus
        ldj = tf.math.reduce_sum(log_sigmas, axis=[1,2,3])
    else:
        z = (x - mus)*tf.math.exp(-log_sigmas)
        ldj = -tf.math.reduce_sum(log_sigmas, axis=[1,2,3])
    return z, ldj
        
class Gaussianize(Parameterize):
    """
    Implementation of parameterize for a Gaussian prior. Corresponds to the "Gaussianization" step in Glow (Kingma et al, 2018).
    """
    def __init__(self, input_shape=None, name='gaussianize', *args, **kwargs):
        super().__init__(*args, num_parameters=2, input_shape=input_shape, name=name, **kwargs)
        
    def _forward(self, x1, x2, **kwargs):
        params = self.parameterizer(x1)
        mus, log_sigmas = params[:,:,:,0::2], params[:,:,:,1::2]
        z2, fldj = gaussianize(x2, mus, log_sigmas)
        return z2, fldj
    
    def _inverse(self, x1, z2, **kwargs):
        params = self.parameterizer(x1)
        mus, log_sigmas = params[:,:,:,0::2], params[:,:,:,1::2]
        x2, ildj = gaussianize(z2, mus, log_sigmas, inverse=tf.constant(True))
        return x2, ildj
    
def log_gaussianize(x, mus, log_sigmas, inverse=tf.constant(False)):
    """
    Standardize log normal random variable x using mus and log_sigmas.
    """
    if inverse:
        scales = tf.math.exp(log_sigmas)
        log_x = tf.math.log(x)
        ldj = log_x
        log_y = log_x*scales + mus
        ldj += log_sigmas
        z = tf.math.exp(log_y)
        return z, ldj
    else:
        scales = tf.math.exp(-log_sigmas)
        log_x = tf.math.log(x)
        ldj = -log_x
        log_y = (log_x - mus)*scales
        ldj -= log_sigmas
        z = tf.math.exp(log_y)
        return z, ldj

class LogGaussianize(Parameterize):
    """
    Implementation of Parameterize for a log-Gaussian prior.
    """
    def __init__(self, input_shape=None, epsilon=1.0E-3, name='log_gaussianize', *args, **kwargs):
        super().__init__(*args, num_parameters=2, input_shape=input_shape, name=name, **kwargs)
        self.epsilon = epsilon
        
    def _forward(self, x1, x2, **kwargs):
        """
        A log normal RV X = exp(mu + sigma*Z) where Z ~ N(0,I).
        The forward pass scales to a standard log normal with mu=0, sigma=1 by computing:
        exp(Z) = (X / exp(mu))^(1/sigma)
        """
        params = self.parameterizer(x1)
        mus, log_sigmas = params[:,:,:,0::2], params[:,:,:,1::2]
        # compute softplus activation
        z2, ldj = log_gaussianize(x2, mus, log_sigmas)
        z2 = tf.where(x2 > self.epsilon, z2, x2)
        ldj = tf.where(x2 > self.epsilon, ldj, tf.zeros_like(ldj))
        return z2, tf.math.reduce_sum(ldj, axis=[1,2,3])
    
    def _inverse(self, x1, z2, **kwargs):
        params = self.parameterizer(x1)
        mus, log_sigmas = params[:,:,:,0::2], params[:,:,:,1::2]
        x2, ldj = log_gaussianize(z2, mus, log_sigmas, inverse=tf.constant(True))
        x2 = tf.where(z2 > self.epsilon, x2, z2)
        ldj = tf.where(z2 > self.epsilon, ldj, tf.zeros_like(ldj))
        return x2, tf.math.reduce_sum(ldj, axis=[1,2,3])
    
def half_gaussianize(x, log_sigmas, inverse=tf.constant(False)):
    if inverse:
        z = tf.math.exp(log_sigmas)*x
        ldj = tf.math.reduce_sum(log_sigmas, axis=[1,2,3])
    else:
        z = x*tf.math.exp(-log_sigmas)
        ldj = -tf.math.reduce_sum(log_sigmas, axis=[1,2,3])
    return z, ldj

class HalfGaussianize(Parameterize):
    """
    Implementation of parameterize for a half-Gaussian prior.
    """
    def __init__(self, input_shape=None, name='gaussianize', *args, **kwargs):
        super().__init__(*args, num_parameters=1, input_shape=input_shape, name=name, **kwargs)
        
    def _forward(self, x1, x2, **kwargs):
        log_sigmas = self.parameterizer(x1)
        z2, fldj = half_gaussianize(x2, log_sigmas)
        return z2, fldj
    
    def _inverse(self, x1, z2, **kwargs):
        log_sigmas = self.parameterizer(x1)
        x2, ildj = half_gaussianize(z2, log_sigmas, inverse=tf.constant(True))
        return x2, ildj
    
def exponentiate(x, log_lambdas, inverse=tf.constant(False)):
    if not inverse:
        z = tf.math.exp(log_lambdas)*x
        ldj = tf.math.reduce_sum(log_lambdas, axis=[1,2,3])
    else:
        z = x*tf.math.exp(-log_lambdas)
        ldj = -tf.math.reduce_sum(log_lambdas, axis=[1,2,3])
    return z, ldj

class Exponentiate(Parameterize):
    """
    Implementation of parameterize for an exponetial prior.
    """
    def __init__(self, input_shape=None, name='gaussianize', *args, **kwargs):
        super().__init__(*args, num_parameters=1, input_shape=input_shape, name=name, **kwargs)
        
    def _forward(self, x1, x2, **kwargs):
        log_lambdas = self.parameterizer(x1)
        z2, fldj = exponentiate(x2, log_lambdas)
        return z2, fldj
    
    def _inverse(self, x1, z2, **kwargs):
        log_lambdas = self.parameterizer(x1)
        x2, ildj = exponentiate(z2, log_lambdas, inverse=tf.constant(True))
        return x2, ildj
