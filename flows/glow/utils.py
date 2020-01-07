import tensorflow as tf

def tf_init_var(event_ndims=1, batch_ndims=1, unspecified_axes=[-1], dtype=tf.float32, **kwargs):
    """
    Creates a new, uninitialized tensorflow variable with the given unspecified axes.
    All dimensions (specified or not) are initially set to 1. Additional keyword args
    are passed to the tf.Variable constructor.
    """
    base_shape = [1]*(event_ndims + batch_ndims)
    unspecified_axes = [i % len(base_shape) for i in unspecified_axes]
    tensor_shape = tf.TensorShape([None if i in unspecified_axes else d for i, d in enumerate(base_shape)])
    return tf.Variable(tf.zeros(base_shape), shape=tensor_shape, dtype=dtype, **kwargs)
