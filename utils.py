import tensorflow as tf
import numpy as np

def update_metrics(metric_dict, **kwargs):
    for k,v in kwargs.items():
        if k in metric_dict:
            prev, n = metric_dict[k]
            metric_dict[k] = ((v + n*prev) / (n+1), n+1)
        else:
            metric_dict[k] = (v, 0)
            
def var(x: tf.Variable):
    """
    Workaround for Tensorflow bug #32748 (https://github.com/tensorflow/tensorflow/issues/32748)
    Converts tf.Variable to a Tensor via tf.identity to prevent tf.function from erroneously
    keeping weak references to garbage collected variables.
    """
    if tf.__version__ >= "2.1.0":
        return x
    else:
        return tf.identity(x)