import tensorflow as tf
import numpy as np

from tensorflow import layers as tfl
from tensorflow.python.ops.init_ops import Initializer


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


class Selection(Initializer):
    """Initializes a tensor to a random pick of a set of possible values.

    Caution! Does not yet verify the shape of the requested tensor.

    Args:
        values: A list of values/arrays from which the initialized value is picked. If
            the values are scalar, they will be converted into the requested shape of the
            variable.
        dtype: the dtype of the initialized value
    """

    def __init__(self, values, dtype=tf.float32):
        self.values = values
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        if isinstance(self.values[0], (float, int)):
            # The values are scalar and we have to transform them into the given shape
            values = tf.stack(self.values, axis=0)
            # We first set the rank to the requested shape + the first dimension
            new_shape = [len(self.values)]
            new_shape.extend([1 for _ in shape])
            values = tf.reshape(values, new_shape)
            # Now we repeat the scalar value to get the requested shape along the
            # new dimensions
            repetitions = [1]  # don't repeat the set of values
            repetitions.extend(shape)
            values = tf.tile(values, repetitions)
        else:
            values = tf.stack(self.values, axis=0)
        return tf.random_shuffle(tf.cast(values, dtype=dtype))[0]

    def get_config(self):
        return {'values': self.values,
                'dtype': self.dtype}


selection_initializer = Selection


def deconv2d(inputs,
             filters,
             kernel_size,
             strides=(1, 1),
             padding='valid',
             data_format='channels_last',
             activation=None,
             use_bias=False,
             bias_initializer=tf.zeros_initializer(),
             kernel_regularizer=None,
             bias_regularizer=None,
             activity_regularizer=None,
             trainable=True,
             name=None,
             reuse=False,
             batch_normalization=True,
             training=False):
    """Deconvolutional Layer. Upsamples a given image with a bilinear interpolation."""
    # Compute the shape of the kernel on basis of the input tensor.
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    kernel_dims = [kernel_size[0], kernel_size[1], filters, inputs.shape[-1]]

    def _deconv2d(inputs, activation):
        return tfl.conv2d_transpose(
            inputs,
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

    if batch_normalization:
        # Apply batch_normalization after convolution and activation only afterwards
        out = _deconv2d(inputs, None)
        out = tfl.batch_normalization(out, training=training, name=name, reuse=reuse)
        if activation is not None:
            out = activation(out)
    else:
        out = _deconv2d(inputs, activation)
    return out


def conv2d(inputs, filters, kernel_size, batch_normalization=False, training=False,
           **kwargs):
    if batch_normalization:
        # Apply batch_normalization after convolution and activation only afterwards
        activation = kwargs.get('activation', None)
        kwargs.update({'activation': None})
        out = tfl.conv2d(inputs, filters, kernel_size, **kwargs)
        out = tfl.batch_normalization(out, training=training,
                                      reuse=kwargs.get('reuse', False),
                                      name=kwargs.get('name', None))
        if activation is not None:
            out = activation(out)
    else:
        out = tfl.conv2d(inputs, filters, kernel_size, **kwargs)
    return out


def adap_conv(inputs, adapter_inputs, filters, kernel_size, trainable=True,
              name='adap_conv', reuse=False, extra_convolution=True,
              initial_scales=[1, 0.1], initialize_half_zero=False, **kwargs):
    """Adapter of features from convolutional layers for a progressive convolutional
    network.

    Args:
        inputs: the input from the current, new column
        adapter_inputs: a list of tensors from the previous layers of all other columns
        filters: innermost dimension of the output space
        other parameters as for tf.layers.conv2d
        extra_convolution: Bool whether or not convolve adapter-inputs with each other
            before concatenation
        initialize_half_zero: Bool whether or not to initialize the combination kernel
            to initially ignore everything but the adapter_inputs
        Any kwargs are passed through to all used conv2d layers.
    Returns:
        Output of the new column, as defined in https://arxiv.org/pdf/1606.04671.pdf,
            equation 2
    """
    class half_zeros_initializer(Initializer):

        def __init__(self, only_dampened=True):
            self.only_dampened = only_dampened

        def __call__(self, shape, dtype=None, partition_info=None):
            """Initializes the first half of the input channel dim to zero, the second
            half either to identity (if input dim = 2 output dim), otherwise xavier."""
            kernel_h, kernel_w, dim_in, dim_out = shape[0], shape[1], shape[2], shape[3]
            if dtype is None:
                dtype = 'float32'

            assert dim_in % 2 == 0

            zeros = np.zeros((kernel_h, kernel_w, int(dim_in / 2), dim_out))
            xavier = lambda: tf.contrib.layers.xavier_initializer()([kernel_h, kernel_w,
                                                                     int(dim_in / 2),
                                                                     dim_out])
            first_half = (0.1 * xavier()) if self.only_dampened else zeros.copy()
            if dim_in == (2 * dim_out):
                second_half = zeros.copy()
                # find index for kernel center
                kc_h = int(np.floor(kernel_h / 2.0))
                kc_w = int(np.floor(kernel_w / 2.0))
                second_half[kc_h, kc_w, :, :] = np.eye(dim_out)
            else:
                second_half = xavier()
            return tf.cast(tf.concat([first_half, second_half], axis=2), dtype)

        def get_config(self):
            return {}

    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('adapter', reuse=reuse):
            # Each adapter input gets scaled by a trainable factor.
            scale = tf.get_variable('scale', [len(adapter_inputs)],
                                    initializer=selection_initializer(initial_scales),
                                    trainable=trainable)
            scaled_adapter_inputs = tf.concat([scale[i] * adapter_inputs[i]
                                               for i in range(len(adapter_inputs))],
                                              axis=-1)
            if extra_convolution:
                adapter = tfl.conv2d(scaled_adapter_inputs, inputs.shape[-1], [1, 1],
                                     reuse=reuse, name='adapter', trainable=trainable,
                                     padding='same',
                                     activation=kwargs.get('activation', None))
            else:
                adapter = scaled_adapter_inputs
        # Concatenate both parts together.
        together = tf.concat([inputs, adapter], axis=-1)

        if initialize_half_zero:
            kwargs['kernel_initializer'] = half_zeros_initializer()
        out = conv2d(together, filters, kernel_size, name='combination', **kwargs)
    return out


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
