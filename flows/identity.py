import tensorflow as tf
from . import Transform

class Identity(Transform):
    def __init__(self, input_shape=None, name='id', *args, **kwargs):
        super().__init__(*args, input_shape=input_shape, name=name, has_constant_ldj=True, **kwargs)
            
    def _forward(self, z, *args, **kwargs):
        return z, 0.0

    def _inverse(self, z, *args, **kwargs):
        return z, 0.0