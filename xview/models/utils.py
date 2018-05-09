import tensorflow as tf
import functools


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def cross_entropy(log_predictions, labels):
    """Tensorflow calculation of cross-entropy between prediction probabilities and
    labels."""
    #cond = tf.reduce_sum(labels, axis=3) == 1
    #return tf.reduce_sum(tf.where(cond, labels * predictions, tf.zeros_like(labels))) /\
    #    (1e-3 + tf.reduce_sum(tf.to_float(cond)))
    with tf.name_scope('cross_entropy'):
        pixel_cross_entropy = -tf.reduce_sum(labels * log_predictions, axis=-1)
        # now mean over all the labelled pixels in the batch
        return tf.div(tf.reduce_sum(pixel_cross_entropy),
                      1e-20 + tf.reduce_sum(labels))
