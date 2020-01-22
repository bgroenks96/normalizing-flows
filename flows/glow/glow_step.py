import tensorflow as tf
import tensorflow_probability as tfp
from flows import Flow, Transform
from .affine_coupling import AffineCoupling, coupling_nn_glow
from . import InvertibleConv, ActNorm, Squeeze

class GlowStep(Transform):
    def __init__(self, input_shape=None, layer=0, coupling_nn_ctor=coupling_nn_glow(),
                 split_axis=-1, name='glow_step', init_from_data=True,
                 *args, **kwargs):
        act_norm = ActNorm(name=f'{name}_act_norm', init_from_data=init_from_data)
        invertible_conv = InvertibleConv(name=f'{name}_inv_conv')
        affine_coupling = AffineCoupling(nn_ctor=coupling_nn_ctor, name=f'{name}_affine_coupling')
        flow_steps = [act_norm, invertible_conv, affine_coupling]
        self.layer = layer
        self.split_axis = split_axis
        self.should_split = self.layer > 0 and self.split_axis is not None
        self.flow = Flow(flow_steps)
        super().__init__(*args, input_shape=input_shape, name=name, requires_init=True, **kwargs)
        
    def _initialize(self, input_shape):
        if self.should_split:
            axis = self.split_axis % input_shape.rank
            split_size = input_shape[axis] // 2**self.layer
            new_shape = tf.TensorShape((*input_shape[:axis], split_size, *input_shape[axis+1:]))
            self.flow.initialize(new_shape)
        else:
            self.flow.initialize(input_shape)
        
    def _split(self, x):
        c = x.shape[-1]
        c_split = c // 2**self.layer
        assert c_split > 0, f'too few channel dimensions for layer {self.layer}'
        return tf.split(x, [c_split, c - c_split], axis=self.split_axis) if self.should_split else (x, None)
        
    def _concat(self, x1, x2=None):
        assert not self.should_split or x2 is not None, 'x2 must have a value for layer > 0'
        return tf.concat([x1, x2], axis=self.split_axis) if self.should_split else x1
        
    def _forward(self, x):
        x1, x2 = self._split(x)
        y1, fldj = self.flow.forward(x1)
        return self._concat(y1, x2), fldj
    
    def _inverse(self, y):
        y1, y2 = self._split(y)
        x1, ildj = self.flow.inverse(y1)
        return self._concat(x1, y2), ildj
    
    def _regularization_loss(self):
        return self.flow.regularization_loss()
