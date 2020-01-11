import tensorflow as tf
import tensorflow_probability as tfp
from flows.glow import GlowFlow, RegularizedBijector

class Glow(tf.Module):
    def __init__(self, prior_distribution: tfp.distributions.Distribution,
                num_layers=1, depth_per_layer=1,
                optimizer=tf.keras.optimizers.Adam(lr=1.0E-4),
                *glow_args, **glow_kwargs):
        self.glow = GlowFlow(num_layers=num_layers, depth=depth_per_layer, *glow_args, **glow_kwargs)
        self.prior = prior_distribution
        self.target_dist = self.glow(prior_distribution)
        self.optimizer = optimizer
        self.reg_modules = None
        
    def train_batch(self, x):
        with tf.GradientTape() as tape:
            z = self.glow.inverse(x)
            ildj = self.glow.inverse_log_det_jacobian(x, event_ndims=self.prior.event_shape.rank)
            prior_log_probs = self.prior.log_prob(z)
            log_probs = prior_log_probs + ildj
            if not self.reg_modules:
                self.reg_modules = [b for b in self.submodules if isinstance(b, RegularizedBijector)]
            nll_loss = -tf.math.reduce_mean(log_probs)
            reg_loss = [b._regularization_loss() for b in self.reg_modules]
            total_loss = nll_loss + tf.math.reduce_sum(reg_loss)
            gradients = tape.gradient(total_loss, self.trainable_variables)
            gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss, nll_loss, ildj, grad_norm
            
    def train(self, data):
        for x_batch in data:
            loss, nll = self.train_batch(x_batch)
            print(f'loss: {loss}, nll: {nll}')
        return self

def create_glow_estimator(prior_distribution: tfp.distributions.Distribution,
                          num_layers=1, depth_per_layer=1,
                          optimizer=tf.keras.optimizers.Adam(),
                          *glow_args, **glow_kwargs):
    glow = GlowFlow(num_layers=num_layers, depth=depth_per_layer, *glow_args, **glow_kwargs)
    target_dist = glow(prior_distribution)
    keras_modules = [module for module in glow.submodules if isinstance(module, tf.keras.Model)]
    def _glow_model_fn(features, labels, mode, params, config):
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        log_probs = target_dist.log_prob(features)
        nll_loss = -tf.math.reduce_sum(log_probs)
        reg_loss = tf.math.add_n(sum([model.get_losses_for(None) for model in keras_modules]))
        total_loss = nll_loss + reg_loss
        mean_elbo_metric = tf.keras.metrics.Mean(name='mean_elbo')
        mean_elbo_metric.update_state(log_probs)
        train_op = None
        if training:
            update_ops = sum([model.get_updates_for(None) for model in keras_modules])
            minimize_op, _ = optimizer.get_updates(total_loss, glow.trainable_variables)
            train_op = tf.group(minimize_op, update_ops)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            eval_metric_ops={
                "mean_elbo": mean_elbo_metric,
            })
    return tf.estimator.Estimator(model_fn=_glow_model_fn)
