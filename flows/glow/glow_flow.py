import tensorflow as tf
import tensorflow_probability as tfp
from flows import Flow, Transform, Invert
from . import Squeeze, GlowStep, coupling_nn_glow

def GlowFlow(input_shape=None, num_layers=1, depth=1,
             coupling_nn_ctor=coupling_nn_glow(),
             init_from_data=True, name='glow_flow',
             *args, **kwargs):
    """
    Creates a new Glow normalizing flow with the given configuration.
    Input shapes for each GlowStep are inferred from the first call to forward
    or inverse. Input shapes must match for all subsequent calls.
    """
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
    # invert squeeze ops;
    # note that it's important to use the original Squeeze instances due to the inferred padding parameters
    flow_steps += list(reversed([Invert(step) for step in flow_steps if isinstance(step, Squeeze)]))
    return Flow(flow_steps, *args, input_shape=input_shape, name=name, **kwargs)
