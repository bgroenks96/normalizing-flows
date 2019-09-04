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

    def _backward(self):
        if self.pre is not None:
            self.pre._backward()

    def _forward(self, x):
        raise NotImplementedError('missing implementation of _forward')

    def _inverse(self, y):
        raise NotImplementedError('missing implementation of _inverse')

    def _inverse_log_det_jacobian(self, y):
        raise NotImplementedError('missing implementation of _inverse_log_det_jacobian')

class Planar(BaseTransform):
    def __init__(self, input_dims, **kwargs):
        super(Planar, self).__init__(input_dims=input_dims, **kwargs)
        self.d = input_dims
        # parameters u, w, b; u is updated by a custom rule, so we disable gradient computation
        # u is initialized as the unit vector in R^d
        self.u = tf.Variable(np.ones((self.d, 1)) / np.sqrt(self.d), name=f'u_{self.unique_id}',
                             dtype=tf.float32, trainable=False)
        self.w = tf.Variable(np.random.uniform(0., 1., size=(self.d, 1)), name=f'w_{self.unique_id}', dtype=tf.float32)
        self.b = tf.Variable(0.0, name=f'b_{self.unique_id}', dtype=tf.float32)
        # define nonlinearity function
        self.h = lambda x: tf.math.tanh(x)
        self.dh = lambda x: 1.0 - tf.square(tf.tanh(x))

    def _alpha(self):
        wu = tf.matmul(self.w, self.u, transpose_a=True)
        m = -1 + tf.math.log(1.0 + tf.math.exp(wu))
        return m - wu

    def _backward(self):
        # first we apply update rule for u parameter ... (see Rezende et al. 2016)
        alpha = self._alpha()
        alpha_w = alpha*self.w / tf.reduce_sum(self.w**2.0)
        self.u.assign(self.u + alpha_w)
        # ... then apply gradients and backpropagate
        super(Planar, self)._backward()

    def _forward(self, z):
        wz = tf.matmul(z, self.w)
        return z + tf.matmul(self.h(wz + self.b), self.u, transpose_b=True)

    def _inverse(self, y):
        alpha = self._alpha()
        z_para = alpha*self.w / tf.reduce_sum(self.w**2.0)
        wz_para = tf.matmul(self.w, z_para, transpose_a=True)
        z_orth = y - tf.transpose(z_para) - self.h(wz_para + self.b)
        return z_orth + tf.transpose(z_para)

    def _forward_log_det_jacobian(self, z):
        wz = tf.matmul(z, self.w)
        dh_dz = tf.matmul(self.dh(wz + self.b), self.w, transpose_b=True)
        return tf.math.abs(1.0 + tf.matmul(dh_dz, self.u))

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

class Radial(BaseTransform):
    def __init__(self, input_dims, **kwargs):
        super(Radial, self).__init__(input_dims=input_dims, **kwargs)
        self.d = input_dims
        self.z_0 = tf.Variable(np.random.uniform(0.1, 1., size=(1, self.d)), name=f'z_0_{self.unique_id}', dtype=tf.float32)
        self.alpha = tf.Variable(0.01, name=f'alpha_{self.unique_id}', dtype=tf.float32,
                                constraint=lambda x: tf.clip_by_value(x, 0.0, np.infty))
        self.beta = tf.Variable(1.0, name=f'beta_{self.unique_id}', dtype=tf.float32, trainable=False)
        self.r = lambda z: tf.norm(z - self.z_0, axis=-1, keepdims=True) # (B,1)
        self.h = lambda alpha, r: 1.0 / (alpha + r) # (B,1)
        self.dh = lambda alpha, r: -1.0 / tf.math.square(alpha + r) # (B,1)


    def _backward(self):
        m = tf.math.log(1.0 + tf.math.exp(self.beta))
        self.beta.assign(-self.alpha + m)
        super(Radial, self)._backward()

    def _forward(self, z):
        assert z.shape[-1] == self.d
        alpha = self.alpha
        r = self.r(z)
        h = self.h(alpha, r)
        y = z + self.beta*h*(z - self.z_0)
        assert y.shape == z.shape
        return y


    def _inverse(self, y):
        alpha = self.alpha
        beta = self.beta
        yz_norm = tf.norm(y - self.z_0)
        # solving || y - z_0 || = r + (beta*r)/(alpha + r) in terms of r; hopefully it's correct
        r = 0.5*(-tf.math.sqrt(alpha**2.0 + 2.0*alpha*(beta + yz_norm)+(beta - yz_norm)**2.0) - alpha - beta + yz_norm)
        with tf.control_dependencies([
            tf.debugging.assert_all_finite(alpha, 'alpha nan'),
            tf.debugging.assert_all_finite(beta, 'beta nan'),
            tf.debugging.assert_all_finite(self.z_0, 'z_0 nan'),
            tf.debugging.assert_all_finite(yz_norm, 'yz_norm nan'),
            tf.debugging.assert_all_finite(r, 'r nan')]):
            return (y - self.z_0) / (r*(1.0 + beta / (alpha + r)))

    def _forward_log_det_jacobian(self, z):
        assert z.shape[-1] == self.d
        alpha = self.alpha
        r = self.r(z) # (B,1)
        h = self.h(alpha, r) # (B,1)
        beta_h_p1 = 1.0 + self.beta*h # (B,1)
        beta_dh_r = self.beta*self.dh(alpha, r)*r # (B,1)
        return tf.math.pow(beta_h_p1, self.d - 1.)*(1. + beta_h_p1 + beta_dh_r)

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

class Affine(BaseTransform):
    def __init__(self, input_dims=2, **kwargs):
        super(Affine, self).__init__(input_dims=input_dims, **kwargs)
        self.k = input_dims
        self.scope = f'affine_{self.unique_id}'
        self.V = tf.Variable(np.identity(self.k), name=f'V_{self.unique_id}', dtype=tf.float32)
        self.L = tf.Variable(np.random.uniform(0, 1.0, size=(self.k*(self.k+1) // 2)), name=f'L_{self.unique_id}', dtype=tf.float32)
        self.b = tf.Variable(np.random.normal(0, 0.5, size=(self.k,)), name=f'b_{self.unique_id}', dtype=tf.float32)

    def _forward(self, x):
        affine_bijector = tfp.bijectors.Affine(
            scale_tril=tfp.distributions.fill_triangular(self.L),
            scale_perturb_factor=self.V,
            shift=self.b
        )
        return affine_bijector.forward(x)

    def _inverse(self, y):
        affine_bijector = tfp.bijectors.Affine(
            scale_tril=tfp.distributions.fill_triangular(self.L),
            scale_perturb_factor=self.V,
            shift=self.b
        )
        return affine_bijector.inverse(y)

    def _inverse_log_det_jacobian(self, y):
        affine_bijector = tfp.bijectors.Affine(
            scale_tril=tfp.distributions.fill_triangular(self.L),
            scale_perturb_factor=self.V,
            shift=self.b
        )
        return affine_bijector.inverse_log_det_jacobian(y, self.inverse_min_event_ndims)

class PReLU(BaseTransform):
    def __init__(self, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.scope = f'prelu_{self.unique_id}'
        self.alpha = tf.Variable(0.01, name='alpha', dtype=tf.float32)

    def _forward(self, z):
        alpha = tf.abs(self.alpha)
        return tf.where(z >= 0, z, alpha*z)

    def _inverse(self, x):
        alpha = tf.abs(self.alpha)
        return tf.where(x >= 0, x, 1.0 / alpha * x)

    def _inverse_log_det_jacobian(self, y):
        I = tf.ones_like(y)
        alpha = tf.abs(self.alpha)
        inv_jacobian = tf.where(y >= 0, I, I * 1.0 / alpha)
        return tf.reduce_sum(tf.math.log(tf.math.abs(inv_jacobian)))
