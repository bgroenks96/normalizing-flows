import tensorflow as tf
import tensorflow_probability as tfp
from .affine_coupling import AffineCoupling, resnet_glow
from . import InvertibleConv, ActNorm, Squeeze

class GlowStep(tfp.bijectors.Bijector):
    def __init__(self, layer=0, coupling_nn_ctor=resnet_glow(), split_axis=-1, name='glow_step',
                 init_from_data=True, forward_min_event_ndims=1, inverse_min_event_ndims=1,
                 *args, **kwargs):
        super().__init__(forward_min_event_ndims=forward_min_event_ndims,
                         inverse_min_event_ndims=inverse_min_event_ndims,
                         name=name, *args, **kwargs)
        #act_norm = ActNorm(name=f'{self.name}/act_norm', init_from_data=init_from_data)
        batch_norm = tfp.bijectors.BatchNormalization(name=f'{self.name}/batch_norm')
        invertible_conv = InvertibleConv(name=f'{self.name}/inv_conv')
        affine_coupling = AffineCoupling(nn_ctor=coupling_nn_ctor, name=f'{self.name}/affine_coupling')
        flow_steps = [batch_norm, invertible_conv, affine_coupling]
        self.flow = tfp.bijectors.Chain(list(reversed(flow_steps)))
        self.layer = layer
        self.split_axis = split_axis
        
    def _split(self, x):
        c = x.shape[-1]
        c_split = c // 2**self.layer
        assert c_split > 0, f'too few channel dimensions for layer {self.layer}'
        return tf.split(x, [c_split, c - c_split], axis=self.split_axis) if self.layer > 0 else (x, None)
        
    def _concat(self, x1, x2=None):
        assert self.layer == 0 or x2 is not None, 'x2 must have a value for layer > 0'
        return tf.concat([x1, x2], axis=self.split_axis) if self.layer > 0 else x1
        
    def _forward(self, x):
        x1, x2 = self._split(x)
        y1 = self.flow._forward(x1)
        return self._concat(y1, x2)
    
    def _inverse(self, y):
        y1, y2 = self._split(y)
        x1 = self.flow._inverse(y1)
        return self._concat(x1, y2)
    
    def _forward_log_det_jacobian(self, x):
        x1, x2 = self._split(x)
        return self.flow._forward_log_det_jacobian(x1)

    def _inverse_log_det_jacobian(self, y):
        y1, y2 = self._split(y)
        return self.flow._inverse_log_det_jacobian(y1)
