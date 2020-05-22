import tensorflow as tf
import tensorflow_probability as tfp
from normalizing_flows.flows import AmortizedTransform

class TriangularSylvester(AmortizedTransform):
    def __init__(self, flip_z=False, **kwparams):
        """
        Triangular Sylvester flow (T-SNF)
        """
        super().__init__(**kwparams)
        # define nonlinearity function
        self.h = lambda x: tf.math.tanh(x)
        self.dh = lambda x: 1.0 - tf.square(tf.tanh(x))
        # define permutation matrix constructor
        if flip_z:
            self.perm_z = lambda z: tf.reverse(tf.eye(tf.shape(z)[-1], batch_shape=tf.shape(z)[:1]), axis=(1,))
        else:
            self.perm_z = lambda z: tf.eye(tf.shape(z)[-1], batch_shape=tf.shape(z)[:1])

    def _param_count(self, shape):
        d = shape[-1]
        return d**2 + 2*d + d

    @tf.function
    def _parameterize(self, d: tf.Tensor, params: tf.Tensor):
        r_full, diags, b = params[:,:d**2], params[:,d**2:-d], params[:,-d:]
        diag_1, diag_2 = diags[:,:d], diags[:,d:]
        r_full = tf.reshape(r_full, (-1, d, d))
        b  = tf.reshape(b, (-1, 1, d))
        triu_mask = tfp.math.fill_triangular(tf.ones(((d**2 + d) // 2,)), upper=True)
        r1 = r_full * triu_mask
        r2 = tf.transpose(r_full, (0, 2, 1)) * triu_mask
        return r1, r2, diag_1, diag_2, b

    @tf.function
    def _diag_r(self, r, diag):
        r_diag = tf.math.tanh(diag)
        r = tf.linalg.set_diag(r, r_diag, k=1)
        return r, r_diag

    @tf.function
    def _permute(self, z, perm_z):
        return tf.matmul(z, perm_z)

    @tf.function
    def _transform(self, z, r1, r2, b):
        lr1 = tf.transpose(r1, (0, 2, 1)) # (batch_size, d, d)
        lr2 = tf.transpose(r2, (0, 2, 1)) # (batch_size, d, d)
        a = tf.matmul(z, lr2) + b # (batch_size, 1, d)
        z = tf.matmul(self.h(a), lr1) # (batch_size, 1, d)
        return z, a

    @tf.function
    def _log_det_jacobian(self, a, r1_diag, r2_diag):
        diag_j = 1.0 + tf.squeeze(self.dh(a), 1) * r1_diag * r2_diag
        log_diag_j = tf.math.log(tf.math.abs(diag_j))
        log_det_j = tf.reduce_sum(log_diag_j, axis=-1, keepdims=True)
        return log_det_j

    @tf.function
    def _forward(self, z, params: tf.Tensor):
        # set up parameters
        d = tf.shape(z)[1]
        r1, r2, diag_1, diag_2, b = self._parameterize(d, params)
        z = tf.expand_dims(z, axis=1) # (batch_size, 1, d)
        # set amortized diagonals for r1, r2 block
        r1, r1_diag = self._diag_r(r1, diag_1)
        r2, r2_diag = self._diag_r(r2, diag_2)
        # apply permutation to z
        perm_z = self.perm_z(z)
        z_ = self._permute(z, perm_z)
        # compute transformation
        z_, a = self._transform(z_, r1, r2, b)
        # permute z back again and add residual
        z = z + self._permute(z_, perm_z)
        # compute log det jacobian
        ldj = self._log_det_jacobian(a, r1_diag, r2_diag)
        return tf.squeeze(z, axis=1), ldj
