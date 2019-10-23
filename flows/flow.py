import tensorflow_probability as tfp
from .base_transform import BaseTransform, AmortizedTransform

class Flow():
    def __init__(self, transform: BaseTransform):
        self.transform = transform
        self.amortized_params = sum([t.param_count for t in self.transform.bijectors if isinstance(t, AmortizedTransform)])

    def amortize(self, args):
        if isinstance(self.transform, tfp.bijectors.Chain):
            assert isinstance(args, list)
            amortized_transforms = [t for t in self.transform.bijectors if isinstance(t, AmortizedTransform)]
            n_a = len(amortized_transforms)
            assert len(args) == n_a, f'expected args for {n_a} flows, got {len(args)}'
            for i, transform in enumerate(amortized_transforms):
                transform.amortize(args[i])
        elif isinstance(self.transform, AmortizedTransform):
            self.transform.amortize(args)
        else:
            raise Exception('not an amortized transform')

    def transform(dist: tfp.distributions.Distribution):
        return tfp.TransformedDistribution(dist, self.transform)
