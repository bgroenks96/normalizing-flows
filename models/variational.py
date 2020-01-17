import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model

def nll_loss(distribution_fn):
    import tensorflow_probability as tfp
    def nll(y_true, y_pred):
        def log_prob(dist: tfp.distributions.Distribution):
            return dist.log_prob(tf.squeeze(y_true))
        dist = tfp.layers.DistributionLambda(distribution_fn, log_prob)
        nll = -dist(y_pred)
        return tf.reduce_mean(nll, axis=-1)
    return nll

def parameterize(model, distribution_fn):
    model.predict_mean = lambda x: distribution_fn(model.predict(x)).mean()
    model.predict_q = lambda x, q: distribution_fn(model.predict(x)).quantile(q)
    model.sample = lambda x: distribution_fn(model.predict(x)).sample()
    return model
