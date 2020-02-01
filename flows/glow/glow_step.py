import tensorflow as tf
import tensorflow_probability as tfp
from flows import Flow, Transform
from flows.affine import BatchNorm
from . import InvertibleConv, ActNorm, Squeeze, AffineCoupling, coupling_nn_glow

class GlowStep(Transform):
    def __init__(self, input_shape=None, coupling_nn_ctor=coupling_nn_glow(),
                 act_norm=True, name='glow_step',
                 *args, **kwargs):
        norm = ActNorm(name=f'{name}_act_norm') if act_norm else BatchNorm(name=f'{name}_batch_norm')
        invertible_conv = InvertibleConv(name=f'{name}_inv_conv')
        affine_coupling = AffineCoupling(nn_ctor=coupling_nn_ctor, name=f'{name}_affine_coupling')
        flow_steps = [norm, invertible_conv, affine_coupling]
        self.flow = Flow(flow_steps)
        super().__init__(*args, input_shape=input_shape, name=name, requires_init=True, **kwargs)
        
    def _initialize(self, input_shape):
        self.flow.initialize(input_shape)
        
    def _forward(self, x, *args, **kwargs):
        return self.flow.forward(x, *args, **kwargs)
    
    def _inverse(self, y, *args, **kwargs):
        return self.flow.inverse(y, *args, **kwargs)
    
    def _regularization_loss(self):
        return self.flow.regularization_loss()
