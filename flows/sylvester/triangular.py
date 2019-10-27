import tensorflow as tf
from flows import BaseTransform

class TriangularSylvester(BaseTransform):
    def __init__(self, flip_z=False, **kwargs):
        """
        Triangular Sylvester flow (T-SNF)
        """
        super(TriangularSylvester, self).__init__(**kwargs)
        # define nonlinearity function
        self.h = lambda x: tf.math.tanh(x)
        self.dh = lambda x: 1.0 - tf.square(tf.tanh(x))
        # define permutation matrix constructor
        if flip_z:
            self.perm_z = lambda z: tf.reverse(tf.eye(tf.shape(z)[1], batch_shape=tf.shape(z)[:1]), axis=(1,))
        else:
            self.perm_z = lambda z: tf.eye(tf.shape(z)[1], batch_shape=tf.shape(z)[:1])

    def param_count(self, d):
        return d**2 + 2*d

    @tf.function
    def forward(self, z, args: tf.Tensor):
        # set up parameters
        d = tf.shape(z)[1]
        r_full, diags, b = args[:,:d**2], args[:,m**2:-d], args[:,-d:-1]
        diag_1, diag_2 = diags[:d], diags[d:]
        r_full = tf.reshape(r_full, (-1, d, d))
        b  = tf.reshape(b, (-1, 1, d))
        z = tf.expand_dims(z, axis=1) # (batch_size, 1, d)
        diag = tf.constant(range(d))
        triu_mask = tf.constant(np.triu(np.ones((d, d))))
        r1 = r_full * triu_mask
        r2 = tr.transpose(r_full, (0, 2, 1)) * triu_mask
        # set amortized diagonals for r1, r2
        r1_diag = tf.math.tanh(diag_1)
        r2_diag = tf.math.tanh(diag_2)
        tf.linalg.set_diag(r1, r1_diag)
        tf.linalg.set_diag(r2, r2_diag)
        # apply permutation to z
        perm_z = self.perm_z(z)
        z_ = tf.matmul(z, perm_z)
        # compute transformation
        lr1 = tf.transpose(r1, (0, 2, 1)) # (batch_size, d, d)
        lr2 = tf.transpose(r2, (0, 2, 1)) # (batch_size, d, d)
        a = tf.matmul(z_, lr2T) + b # (batch_size, 1, d)
        z_ = tf.matmul(self.h(a), lr1T) # (batch_size, 1, d)
        # permute z back again and add residual
        z = z + tf.matmul(z_, perm_z)
        # compute log det jacobian
        diag_j = 1.0 + tf.squeeze(self.dh(a), 1) * r1_diag * r2_diag
        log_diag_j = tf.math.log(tf.math.abs(diag_j))
        log_det_j = tf.reduce_sum(log_diag_j, axis=-1)
        return tf.squeeze(z, axis=1), log_det_j
