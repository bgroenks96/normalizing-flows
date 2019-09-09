import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class BaseTransform(tfp.bijectors.Bijector, tf.Module):
    unique_id = 0
    def __init__(self, **kwargs):
        min_event_ndims = kwargs['min_event_ndims'] if 'min_event_ndims' in kwargs else 1
        validate_args = kwargs['validate_args'] if 'validate_args' in kwargs else False
        name = kwargs['name'] if 'name' in kwargs else type(self).__name__
        super(BaseTransform, self).__init__(
            forward_min_event_ndims=min_event_ndims,
            inverse_min_event_ndims=min_event_ndims,
            validate_args=validate_args,
            name=name)
        self.pre = kwargs['pre'] if 'pre' in kwargs else None
        self.kwargs = kwargs
        self.unique_id = BaseTransform.unique_id
        BaseTransform.unique_id += 1

    def __call__(self, input_transform):
        kwargs = dict(self.kwargs)
        kwargs['pre'] = input_transform
        return self.__class__(**kwargs)

    def _forward(self, x):
        raise NotImplementedError('missing implementation of _forward')

    def _inverse(self, y):
        raise NotImplementedError('missing implementation of _inverse')

    def _inverse_log_det_jacobian(self, y):
        raise NotImplementedError('missing implementation of _inverse_log_det_jacobian')


class Flow():
    def __init__(self, base_dist: tfp.distributions.Distribution, transform: BaseTransform, name: str,
                 input_shape=None, learning_rate=1.0E-3):
        bijectors = []
        trainable_vars = []
        next = transform
        while next is not None:
            bijectors.append(next)
            trainable_vars += next.trainable_variables
            next = next.pre
        chain = tfp.bijectors.Chain(bijectors, name=name)
        self.transform = transform
        self.dist = tfp.distributions.TransformedDistribution(distribution=base_dist, bijector=chain)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.trainable_variables = trainable_vars

    @tf.function
    def train_on_batch(self, X):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.dist.log_prob(X))
            grads = tape.gradient(loss, self.trainable_variables)
            grads = [tf.clip_by_value(grad, -10, 10) for grad in grads]
            with tf.control_dependencies([tf.debugging.assert_all_finite(grad, f'nan/inf gradient for {var.name}') for grad, var in zip(grads, self.trainable_variables)]):
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                return loss, grads
