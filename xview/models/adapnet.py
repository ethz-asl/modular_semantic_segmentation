import tensorflow as tf
from tensorflow.python.layers.layers import max_pooling2d

from .base_model import BaseModel
from .custom_layers import conv2d, deconv2d, log_softmax
from .utils import cross_entropy


# Definition of the different building blocks from the paper.


def block_a(inputs, intermed_filters, filters, strides, is_training, name, reuse,
            activation, shortcut_conv=False):
    """
    Block A of the Adapnet/Resnet architecture.

    Args:
        inputs: The input tensor.
        intermed_filters: dimension of intermediate features
        filters: dimension of block-output features
        strides: strides applied to the first convolution (and possibly the shortcut
            convolution)
        is_training (bool): Indicator whether batch_normalization should be in training
            (batch) or testing (continuous) mode.
        name: variable-scope for this block
        reuse (bool): If true, reuse existing variables of same name
            (attention with prefix). Will raise error if it cannot find such variables.
        activation: activation function applied to the block output
        shortcut_conv (bool): If True, a convolution is applied in the shortcut branch.
            Has to be set to True in case input- and output-dimensions are not equal.
    Returns:
        block output tensor
    """
    # Common parameters of the convolutions
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': True, 'training': is_training, 'use_bias': False}

    with tf.variable_scope(name):
        stage_1 = conv2d(inputs, intermed_filters, 1, strides=strides, name='stage_1',
                         **params)
        stage_2 = conv2d(stage_1, intermed_filters, 3, name='stage_2', **params)
        stage_3 = conv2d(stage_2, filters, 1, name='stage_3', **params)

        if shortcut_conv:
            shortcut = conv2d(inputs, filters, 1, strides=strides, name='shortcut',
                              **params)
        else:
            shortcut = tf.identity(inputs)
    return activation(tf.add(stage_3, shortcut))


def block_b(inputs, filters_1, filters_2, filters_3, dilation1, dilation2, is_training,
            name, reuse, activation, shortcut_conv=False):
    """
    Block B of the Adapnet/Resnet architecture.

    Args:
        inputs: The input tensor.
        filters_1: dimension of features after first convolution
        filters_2: dimension of features after second convolution
        filters_3: dimension of features after third/last convolution (i.e. = output)
        dilation_1: Dilation rate for the first branch of the second convolution
        dilation_1: Dilation rate for the second branch of the second convolution
        is_training (bool): Indicator whether batch_normalization should be in training
            (batch) or testing (continuous) mode.
        name: variable-scope for this block
        reuse (bool): If true, reuse existing variables of same name
            (attention with prefix). Will raise error if it cannot find such variables.
        activation: activation function applied to the block output
        shortcut_conv (bool): If True, a convolution is applied in the shortcut branch.
            Has to be set to True in case input- and output-dimensions are not equal.
    Returns:
        block output tensor
    """
    # Common parameters of the convolutions
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': True, 'training': is_training, 'use_bias': False}

    with tf.variable_scope(name):
        stage_1 = conv2d(inputs, filters_1, 1, name='stage_1', **params)
        # As an adaption to regular ResNet, the stage 2 gets split up into 2 atrous
        # convolutions with different dilration rate and then concatenated together.
        stage_2_1 = conv2d(stage_1, filters_2 / 2, 3, dilation_rate=dilation1,
                           name='stage_2_1', **params)
        stage_2_2 = conv2d(stage_1, filters_2 / 2, 3, dilation_rate=dilation2,
                           name='stage_2_2', **params)
        stage_2 = tf.concat([stage_2_1, stage_2_2], axis=3)
        stage_3 = conv2d(stage_2, filters_3, 1, name='stage_3', **params)

        if shortcut_conv:
            shortcut = conv2d(inputs, filters_3, 1, name='shortcut', **params)
        else:
            shortcut = tf.identity(inputs)
    return activation(tf.add(stage_3, shortcut))


def adapnet(inputs, prefix, num_units, num_classes, is_training=False, reuse=True):
    """
    Adapnet Architecture, as proposed in
    http://ais.informatik.uni-freiburg.de/publications/papers/valada17icra.pdf

    Args:
        inputs: The input tensor
        prefix: Name prefix applied to all variables
        num_units: Number of feature units before the final deconvolution
        num_classes: Number of output classes
        is_training (bool): Indicator whether batch_normalization should be in training
            (batch) or testing (continuous) mode.
        reuse (bool): If true, reuse existing variables of same name
            (attention with prefix). Will raise error if it cannot find such variables.
    Returns:
        Dict of (also intermediate) block outputs
    """
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': True, 'training': is_training}
    block_params = {'activation': tf.nn.relu, 'reuse': reuse, 'is_training': is_training}
    with tf.variable_scope(prefix):
        block_0_1 = conv2d(inputs, 64, 3, name='block_0_1', **params)
        block_0_2 = conv2d(block_0_1, 64, 7, strides=2, name='block_0_2', **params)
        block_0_pool = max_pooling2d(block_0_2, [2, 2], [2, 2], name='block_0_pool')

        block_1 = block_a(block_0_pool, 64, 256, 1, name='block_layer_1',
                          shortcut_conv=True, **block_params)
        block_2 = block_a(block_1, 64, 256, 1, name='block_layer_2', **block_params)
        block_3 = block_a(block_2, 64, 256, 1, name='block_layer_3', **block_params)
        block_4 = block_a(block_3, 128, 512, 2, name='block_layer_4', shortcut_conv=True,
                          **block_params)
        block_5 = block_a(block_4, 128, 512, 1, name='block_layer_5', **block_params)
        block_6 = block_a(block_5, 128, 512, 1, name='block_layer_6', **block_params)
        block_7 = block_b(block_6, 128, 64, 512, 1, 2, name='block_layer_7',
                          **block_params)
        shortcut = conv2d(block_7, num_units, 1, name='shortcut',
                          activation=None, padding='same', reuse=reuse,
                          batch_normalization=True, training=is_training)

        block_8 = block_a(block_7, 256, 1024, 2, name='block_layer_8',
                          shortcut_conv=True, **block_params)
        block_9 = block_a(block_8, 256, 1024, 1, name='block_layer_9', **block_params)
        block_10 = block_b(block_9, 256, 256, 1024, 1, 2, name='block_layer_10',
                           **block_params)
        block_11 = block_b(block_10, 256, 256, 1024, 1, 4, name='block_layer_11',
                           **block_params)
        block_12 = block_b(block_11, 256, 256, 1024, 1, 8, name='block_layer_12',
                           **block_params)
        block_13 = block_b(block_12, 256, 256, 1024, 1, 16, name='block_layer_13',
                           **block_params)
        block_14 = block_b(block_13, 512, 512, 2048, 2, 4, name='block_layer_14',
                           shortcut_conv=True, **block_params)
        block_15 = block_b(block_14, 512, 512, 2048, 2, 8, name='block_layer_15',
                           **block_params)
        block_16 = block_b(block_15, 512, 512, 2048, 2, 16, name='block_layer_16',
                           **block_params)
        deconv_1 = conv2d(block_16, 2048, 1, name='first_deconvolution_conv', **params)
        deconv_1 = deconv2d(deconv_1, num_units, 4, strides=2, activation=None,
                            reuse=reuse, name='first_deconvolution_upconv',
                            padding='same', batch_normalization=True,
                            training=is_training)

        merge = tf.add(deconv_1, shortcut)
        score = deconv2d(merge, num_classes, 16, strides=8, activation=None,
                         reuse=reuse, name='second_deconvolution_upconv',
                         padding='same', batch_normalization=True,
                         training=is_training)
    return {
        'block_0_1': block_0_1, 'block_0_2': block_0_2, 'block_0_pool': block_0_pool,
        'block_1': block_1, 'block_2': block_2, 'block_3': block_3, 'block_4': block_4,
        'block_5': block_5, 'block_6': block_6, 'block_7': block_7,
        'shortcut': shortcut,
        'block_8': block_8, 'block_9': block_9, 'block_10': block_10,
        'block_11': block_11, 'block_12': block_12, 'block_13': block_13,
        'block_14': block_14, 'block_15': block_15, 'block_16': block_16,
        'deconv_1': deconv_1, 'merge': merge, 'score': score}


class Adapnet(BaseModel):

    def __init__(self, data_description, prefix=None, output_dir=None, **config):
        standard_config = {
            'train_encoder': True
        }
        standard_config.update(config)

        if prefix is None:
            self.prefix = config['modality']
        else:
            self.prefix = prefix

        BaseModel.__init__(self, data_description, output_dir=output_dir,
                           **standard_config)

    def _build_graph(self, train_data, test_data):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Training network
        train_x = train_data[self.config['modality']]
        # ground truth labels
        train_y = tf.to_float(train_data['labels'])

        adapnet_layers = adapnet(train_x, self.prefix, self.config['num_units'],
                                 self.config['num_classes'], is_training=True,
                                 reuse=False)
        with tf.variable_scope('softmax_loss'):
            prob = log_softmax(adapnet_layers['score'], self.config['num_classes'],
                               name='prob')
            self.loss = tf.div(tf.reduce_sum(cross_entropy(prob, train_y)),
                               tf.reduce_sum(train_y))

        # Network for testing / evaluation
        # rgb channel
        test_x = test_data[self.config['modality']]
        adapnet_layers = adapnet(test_x, self.prefix, self.config['num_units'],
                                 self.config['num_classes'], is_training=False,
                                 reuse=True)
        self.prob = tf.nn.softmax(adapnet_layers['score'], name='prob_normalized')
        self.prediction = tf.argmax(self.prob, 3, name='label_2d')

        # Add summaries for some weights
        variable_names = []
        for name in variable_names:
            var = next(v for v in tf.global_variables() if v.name == name)
            tf.summary.histogram(name, var)
