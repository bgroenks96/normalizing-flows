import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D
from flows import Transform

class Parameterize(Transform):
    """
    Generalized base type for parameterizing a pre-specified density given some factored out latent variables.
    """
    def __init__(self, num_parameters, input_shape=None, name='parameterize', *args, **kwargs):
        """
        Base class constructor. Should not be directly invoked by callers.
        
        num_parameters : number of distribution parameters per channel dimension (e.g. 2 for a Gaussian, mu and sigma)
        """
        self.num_parameters = num_parameters
        self.parameterizer = None
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)
        
    def _build_parameterizer_fn(self, z_shape):
        """
        Builds a simple, convolutional neural network for parameterizing a distribution
        with 'num_parameters' parameters. Can be overridden by subclasses.
        """
        x = Input(z_shape[1:])
        h = Conv2D(self.num_parameters*z_shape[-1], 3, padding='same', activation='linear', kernel_initializer='zeros')(x)
        params = Conv2D(self.num_parameters*z_shape[-1], 1, activation='linear', kernel_initializer='zeros')(h)
        return Model(inputs=x, outputs=params)
        
    def _initialize(self, input_shape):
        if self.parameterizer is None:
            #self.log_scale = tf.Variable(tf.zeros((1,1,1,self.num_parameters*input_shape[-1])), name=f'{self.name}/log_scale')
            self.parameterizer = self._build_parameterizer_fn(input_shape)
            
    def _forward(self, x1, x2, **kwargs):
        raise NotImplementedError('missing implementation for Parameterize::_forward')
    
    def _inverse(self, x1, z2, **kwargs):
        raise NotImplementedError('missing implementation for Parameterize::_inverse')
        
class Gaussianize(Parameterize):
    """
    Implementation of parameterize for a Gaussian prior. Corresponds to the "Gaussianization" step in Glow (Kingma et al, 2018).
    """
    def __init__(self, input_shape=None, name='gaussianize', *args, **kwargs):
        super().__init__(*args, num_parameters=2, input_shape=input_shape, name=name, **kwargs)
        
    def _forward(self, x1, x2, **kwargs):
        params = self.parameterizer(x1)#*tf.math.exp(self.log_scale)
        mus, log_sigmas = params[:,:,:,0::2], params[:,:,:,1::2]
        z2 = (x2 - mus)*tf.math.exp(-log_sigmas)
        fldj = -tf.math.reduce_sum(log_sigmas, axis=[1,2,3])
        return z2, fldj
    
    def _inverse(self, x1, z2, **kwargs):
        params = self.parameterizer(x1)#*tf.math.exp(self.log_scale)
        mus, log_sigmas = params[:,:,:,0::2], params[:,:,:,1::2]
        x2 = tf.math.exp(log_sigmas)*z2 + mus
        ildj = tf.math.reduce_sum(log_sigmas, axis=[1,2,3])
        return x2, ildj