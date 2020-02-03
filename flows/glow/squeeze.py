import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows import Transform

class Squeeze(Transform):
    def __init__(self, input_shape=None, factor=2, *args, **kwargs):
        """
        Creates a new transform for the "squeeze" operation, where spatial dimensions are folded
        into channel dimensions. This bijector requires the input data to be 3-dimensional,
        height-width-channel (HWC) formatted images (exluding the batch axis). This implementation
        provides automatic padding/cropping for image sizes not divisible by factor.
        """
        self.factor = factor
        self.padding_x = None
        self.padding_y = None
        super().__init__(*args,
                         input_shape=input_shape,
                         requires_init=True,
                         has_constant_jacobian=True,
                         **kwargs)
        
    def _initialize(self, shape):
        if self.padding_x is None or self.padding_y is None:
            assert shape.rank == 4, f'input should be 4-dimensional, got {shape}'
            batch_size, ht, wt, c = shape[0], shape[1], shape[2], shape[3]
            self.padding_y, self.padding_x = ht % self.factor, wt % self.factor

    def _forward(self, x, *args, **kwargs):
        shape = x.shape
        factor = self.factor
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
        return y, 0.0

    def _inverse(self, y, *args, **kwargs):
        shape = y.shape
        factor = self.factor
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
        return x, 0.0
    
    def _forward_shape(self, shape):
        assert self.input_shape is not None, 'not initialized'
        factor = self.factor
        return tf.TensorShape((shape[0], np.ceil(shape[1] / factor), np.ceil(shape[2] / factor), shape[3]*factor*factor))
    
    def _inverse_shape(self, shape):
        assert self.input_shape is not None, 'not initialized'
        factor = self.factor
        pad_x, pad_y = self.padding_x, self.padding_y
        return tf.TensorShape((shape[0], shape[1]*factor+pad_x, shape[2]*factor+pad_y, shape[3]//factor//factor))
