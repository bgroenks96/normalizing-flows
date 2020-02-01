import tensorflow as tf
import tensorflow_probability as tfp
from flows import Flow, Transform, Invert
from . import Squeeze, Split, Parameterize, GlowStep, coupling_nn_glow

class GlowLayer(Transform):
    def __init__(self,
                 parameterize: Parameterize,
                 input_shape=None,
                 depth=4,
                 coupling_nn_ctor=coupling_nn_glow(),
                 split_axis=-1,
                 act_norm=True,
                 name='glow_layer',
                 *args, **kwargs):
        squeeze = Squeeze(name=f'{name}_squeeze')
        steps = Flow.uniform(depth, lambda i: GlowStep(coupling_nn_ctor=coupling_nn_ctor,
                                                       act_norm=act_norm,
                                                       name=f'{name}_step{i}'))
        layer_steps = [squeeze, steps]
        if split_axis is not None:
            layer_steps.append(Split(parameterize, split_axis=split_axis, name=f'{name}_split'))
        self.flow = Flow(layer_steps)
        super().__init__(*args, input_shape=input_shape, name=name, **kwargs)
        
    def _forward_shape(self, input_shape):
        return self.flow._forward_shape(input_shape)
    
    def _inverse_shape(self, input_shape):
        return self.flow._inverse_shape(input_shape)
        
    def _initialize(self, input_shape):
        self.flow.initialize(input_shape)
        
    def _forward(self, x, **kwargs):
        return self.flow.forward(x, **kwargs)
    
    def _inverse(self, y, *args, **kwargs):
        return self.flow.inverse(y, *args, **kwargs)
    
    def _regularization_loss(self):
        return self.flow._regularization_loss()