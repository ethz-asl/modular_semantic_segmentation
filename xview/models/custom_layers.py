import tensorflow as tf
import numpy as np

from tensorflow.python.layers.layers import conv2d_transpose


def bilinear_filter_initializer(filter_shape):
    """From http://cv-tricks.com/image-segmentation/
    transpose-convolution-in-tensorflow/ and the code of DA-RNN.
    filter_shape is [width, height, num_in_channels, num_out_channels] """
    # Centre location of the filter for which value is calculated
    width = filter_shape[0]
    height = filter_shape[1]
    factor = np.ceil(width/2.0)
    center = (2 * factor - 1 - factor % 2) / (2.0 * factor)
    bilinear = np.zeros([width, height])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / factor - center)) * (1 - abs(y / factor - center))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    return tf.constant_initializer(value=weights, dtype=tf.float32, verify_shape=True)


def deconv2d(inputs,
             filters,
             kernel_size,
             strides=(1, 1),
             padding='valid',
             data_format='channels_last',
             activation=None,
             use_bias=True,
             bias_initializer=tf.zeros_initializer(),
             kernel_regularizer=None,
             bias_regularizer=None,
             activity_regularizer=None,
             trainable=True,
             name=None,
             reuse=None):
    # Compute the shape of the kernel on basis of the input tensor.
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    kernel_dims = [kernel_size[0], kernel_size[1], filters, inputs.shape[-1]]

    return conv2d_transpose(inputs,
                            filters,
                            kernel_size,
                            strides=strides,
                            padding=padding,
                            data_format=data_format,
                            activation=activation,
                            use_bias=use_bias,
                            kernel_initializer=bilinear_filter_initializer(kernel_dims),
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            activity_regularizer=activity_regularizer,
                            trainable=trainable,
                            name=name,
                            reuse=reuse)


def log_softmax(inputs, num_classes, name=None):
    """Log-softmax as defined in DA-RNN code. Not sure why they do not use the standard
    softmax."""
    with tf.name_scope(name, default_name='log_softmax', values=[inputs]):
        input_shape = inputs.get_shape()
        ndims = input_shape.ndims
        array = np.ones(ndims)
        array[-1] = num_classes

        m = tf.reduce_max(inputs, reduction_indices=[ndims-1], keep_dims=True)
        multiples = tf.convert_to_tensor(array, dtype=tf.int32)
        d = tf.subtract(inputs, tf.tile(m, multiples))
        e = tf.exp(d)
        s = tf.reduce_sum(e, reduction_indices=[ndims-1], keep_dims=True)
        return tf.subtract(d, tf.log(tf.tile(s, multiples)))


def softmax(inputs, num_classes, name=None):
    """Softmax as defined in DA-RNN code. Not sure why they do not use the standard
    softmax."""
    with tf.name_scope(name, default_name='softmax', values=[inputs]):
        input_shape = inputs.get_shape()
        ndims = input_shape.ndims
        array = np.ones(ndims)
        array[-1] = num_classes

        m = tf.reduce_max(inputs, reduction_indices=[ndims-1], keep_dims=True)
        multiples = tf.convert_to_tensor(array, dtype=tf.int32)
        e = tf.exp(tf.subtract(inputs, tf.tile(m, multiples)))
        s = tf.reduce_sum(e, reduction_indices=[ndims-1], keep_dims=True)
        return tf.div(e, tf.tile(s, multiples))
