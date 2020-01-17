import tensorflow as tf
import tensorflow_probability as tfp
from flows.glow import GlowFlow, RegularizedBijector

class Glow(tf.Module):
    def __init__(self,
                 prior,
                 prior_parameterize=None,
                 num_layers=1, depth_per_layer=1,
                 optimizer=tf.keras.optimizers.Adam(lr=1.0E-3),
                 clip_grads=True,
                 *glow_args, **glow_kwargs):
        """
        Creates a new Glow model for variational inference.
        
        prior : either a tfp.distributions.Distribution instance that represents the variational prior,
                or a function f: (X -> Theta) where X is the model inputs and Theta is a tensor of distribution parameters.
                In the latter case, prior_parameterize must also be provided.
        prior_parameterize : a function f: (Theta -> tfp.distributions.Distribution) which parameterizes a distribution
                             from the parameter values produced by 'prior'
        num_layers : number of "layers" in the Glow flow; see Kingma and Dhariwal 2018
        depth_per_layer : number of Glow steps per layer; see Kingma and Dhariwal 2018
        optimizer : optimizer to use during training
        clip_grads : True if gradients should be clipped to the global norm, False otherwise
        glow_args : additional arguments for GlowFlow
        glow_kwargs : additional keyword arguments for GlowFlow
        """
        if not isinstance(prior, tfp.distributions.Distribution):
            assert prior_parameterize is not None, 'prior_parameterize must be provided for conditional priors'
        self.glow = GlowFlow(num_layers=num_layers, depth=depth_per_layer, *glow_args, **glow_kwargs)
        self.prior = prior
        self.prior_parameterize = prior_parameterize
        self.optimizer = optimizer
        self.clip_grads = clip_grads
        self.reg_modules = None
        
    def train_batch(self, x):
        """
        Performs a single iteration of mini-batch SGD on input x.
        Returns loss, nll, prior, ildj[, grad_norm]
                where loss is the total optimized loss (including regularization),
                nll is the averaged negative log likelihood component,
                prior is the averaged prior negative log likelihodd,
                ildj is the inverse log det jacobian,
                and, if clip_grads is True, grad_norm is the global max gradient norm
        """
        with tf.GradientTape() as tape:
            if isinstance(self.prior, tfp.distributions.Distribution):
                prior = self.prior
            else:
                prior_params = self.prior(x)
                prior = self.prior_parameterize(prior_params)
            z = self.glow.inverse(x)
            prior_log_probs = tf.math.reduce_sum(prior.log_prob(z), axis=[1,2,3])
            ildj = self.glow.inverse_log_det_jacobian(x, event_ndims=x.shape.rank-1)
            # clip ILDJ to have same norm as prior log probs; this is for optimization purposes only!
            ildj = tf.clip_by_norm(ildj, tf.norm(prior_log_probs))
            log_probs = prior_log_probs + ildj
            if not self.reg_modules:
                self.reg_modules = [b for b in self.submodules if isinstance(b, RegularizedBijector)]
            nll_loss = -tf.math.reduce_mean(log_probs)
            reg_loss = [b._regularization_loss() for b in self.reg_modules]
            total_loss = nll_loss + tf.math.reduce_sum(reg_loss)
            gradients = tape.gradient(total_loss, self.trainable_variables)
            if self.clip_grads:
                gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        if self.clip_grads:
            return total_loss, nll_loss, -tf.math.reduce_mean(prior_log_probs), tf.math.reduce_mean(ildj), grad_norm
        else:
            return total_loss, nll_loss, -tf.math.reduce_mean(prior_log_probs), tf.math.reduce_mean(ildj)
            
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
