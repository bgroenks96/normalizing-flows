import tensorflow as tf
import tensorflow_probability as tfp
from flows import Flow, Transform, Invert
from . import Squeeze, GlowStep, coupling_nn_glow

def GlowFlow(input_shape=None, num_layers=1, depth=1,
             coupling_nn_ctor=coupling_nn_glow(),
             init_from_data=True,
             invert=True,
             name='glow_flow',
             *args, **kwargs):
    """
    Creates a new Glow normalizing flow with the given configuration.
    Note that all Glow ops define forward as x -> z (data to encoding)
    rather than the canonical interpretation of z -> z'. Conversely,
    inverse is defined as z -> x (encoding to data). The implementations
    provided by this module are written to be consistent with the
    terminology as defined by the Glow authors. However, to be consistent
    with typical use of the 'flows' module, the flow is inverted by default.
    
    input_shape : shape of input; can be provided here or at a later time to 'initialize'
    num_layers  : number of "layers" in the multi-scale Glow architecture
    depth       : number of glow steps per layer
    coupling_nn_ctor : function that constructs a Keras model for affine coupling steps
    invert      : whether or not the flow direction should be inverted; defaults to True
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
    flow = Flow(flow_steps, *args, input_shape=input_shape, name=name, **kwargs)
    return Invert(flow) if invert else flow
