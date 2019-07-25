import tensorflow as tf
import tensorflow_probability as tfp
from transforms import BaseTransform

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
        self.dist = tfp.distributions.TransformedDistribution(distribution=base_dist, bijector=chain)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.trainable_variables = trainable_vars

    @tf.function
    def train_on_batch(self, X):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.dist.log_prob(X))
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss, grads
