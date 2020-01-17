import tensorflow as tf
import tensorflow_probability as tfp

class Squeeze(tfp.bijectors.Bijector):
    def __init__(self, factor=2,
                 forward_min_event_ndims=0, inverse_min_event_ndims=0,
                 *args, **kwargs):
        """
        Creates a new bijector for the "squeeze" operation, where spatial dimensions are folded
        into channel dimensions. This bijector requires the input data to be 3-dimensional,
        height-width-channel (HWC) formatted images (exluding the batch axis).
        """
        super().__init__(forward_min_event_ndims=forward_min_event_ndims,
                         inverse_min_event_ndims=inverse_min_event_ndims,
                         *args, **kwargs)
        self.factor = factor
        self.padding_x = None
        self.padding_y = None
        
    def _init_vars(self, input_shape):
        if self.padding_x is None or self.padding_y is None:
            batch_size, ht, wt, c = input_shape
            self.padding_y, self.padding_x = ht % self.factor, wt % self.factor

    def _forward(self, x):
        self._init_vars(x.shape)
        shape = x.shape
        factor = self.factor
        assert shape.rank == 4, 'input should be 4-dimensional'
        h, w, c = shape[1:]
        # pad to divisor
        x = tf.image.resize_with_crop_or_pad(x, h+self.padding_y, w+self.padding_x)
        shape = x.shape
        h, w, c = shape[1:]
        # reshape to intermediate tensor
        x_ = tf.reshape(x, (-1, h // factor, factor, w // factor, factor, c))
        # transpose factored out dimensions to channel axis
        x_ = tf.transpose(x_, [0, 1, 3, 5, 2, 4])
        # reshape to final output shape
        y = tf.reshape(x_, (-1, h // factor, w // factor, c*factor*factor))
        return y

    def _inverse(self, y):
        self._init_vars(y.shape)
        shape = y.shape
        factor = self.factor
        assert shape.rank == 4, 'input should be 4-dimensional'
        h, w, c = shape[1:]
        c_factored = c // factor // factor
        # reshape to intermediate tensor
        y_ = tf.reshape(y, (-1, h, w, c_factored, factor, factor))
        # transpose factored out dimensions back to original intermediate axes
        y_ = tf.transpose(y_, [0, 1, 4, 2, 5, 3])
        # reshape to final output shape
        x = tf.reshape(y_, (-1, h*factor, w*factor, c_factored))
        # crop out padding
        x = tf.image.resize_with_crop_or_pad(x, h*factor-self.padding_y, w*factor-self.padding_x)
        return x
    
    def _forward_log_det_jacobian(self, x):
        return tf.constant(0.0, dtype=x.dtype)
    
    def _inverse_log_det_jacobian(self, y):
        return tf.constant(0.0, dtype=y.dtype)
