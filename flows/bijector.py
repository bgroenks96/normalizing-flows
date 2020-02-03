import tensorflow_probability as tfp
from flows import Transform

class TransformBijector(tfp.bijectors.Bijector):
    """
    Adapter to allow Transforms to be used as TFP Bijectors.
    """
    def __init__(self,
                 transform: Transform,
                 forward_min_event_ndims=0,
                 inverse_min_event_ndims=0,
                 name='transform'):
        super().__init__(forward_min_event_ndims=forward_min_event_ndims,
                         inverse_min_event_ndims=inverse_min_event_ndims,
                         is_constant_jacobian=transform.has_constant_ldj,
                         name=name)
        self.transform = transform
        
    def _forward(self, x):
        y, _ = self.transform.forward(x)
        return y
    
    def _inverse(self, y):
        x, _ = self.transform.inverse(y)
        return x
    
    def _forward_log_det_jacobian(self, x):
        _, fldj = self.transform.forward(x)
        return fldj
    
    def _inverse_log_det_jacobian(self, y):
        _, ildj = self.transform.inverse(y)
        return ildj
    