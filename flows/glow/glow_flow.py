import tensorflow as tf
import tensorflow_probability as tfp
from . import Squeeze, GlowStep, resnet_glow

class GlowFlow(tfp.bijectors.Bijector):
    def __init__(self, num_layers=1, depth=1, coupling_nn_ctor=resnet_glow(),
                 init_from_data=True, name='glow_flow',
                 forward_min_event_ndims=0, inverse_min_event_ndims=0,
                 *args, **kwargs):
        """
        Creates a new Glow normalizing flow bijector with the given configuration.
        Input shapes for each GlowStep are inferred from the first call to forward
        or inverse. Input shapes must match for all subsequent calls.
        """
        super().__init__(forward_min_event_ndims=forward_min_event_ndims,
                         inverse_min_event_ndims=inverse_min_event_ndims,
                         name=name, *args, **kwargs)
        assert num_layers > 0
        assert depth > 0
        flow_steps = []
        for layer in range(num_layers):
            squeeze = Squeeze(name=f'squeeze{layer}')
            flow_steps.append(squeeze)
            for j in range(depth):
                glow_step = GlowStep(layer=layer, coupling_nn_ctor=coupling_nn_ctor,
                                     init_from_data=init_from_data,
                                     name=f'glow_layer{layer}_step{j}')
                flow_steps.append(glow_step)
        # invert squeeze ops and reverse ordering;
        # note that it's important to use the original Squeeze instances due
        # to the inferred padding parameters
        flow_steps += list(reversed([tfp.bijectors.Invert(step) for step in flow_steps if isinstance(step, Squeeze)]))
        self.flow = tfp.bijectors.Chain(list(reversed(flow_steps)))
       
    def _forward(self, x):
        # The forward pass of Glow is designed to go from inputs x -> encoding z;
        # In TFP, the forward direction is from the encoded distribution z -> x;
        # So we invert the directions to allow Glow to be used as a standard TFP bijector
        return self.flow._inverse(x)
    
    def _inverse(self, y):
        # apply forward flow for inverse
        x = self.flow._forward(y)
        return x
    
    def _forward_log_det_jacobian(self, x):
        # apply inverse ldj for forward
        return self.flow._inverse_log_det_jacobian(x)
    
    def _inverse_log_det_jacobian(self, y):
        # apply forward ldj for inverse
        return self.flow._forward_log_det_jacobian(y)
