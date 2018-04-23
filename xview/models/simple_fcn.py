import tensorflow as tf
from tensorflow.python.layers.layers import dropout, max_pooling2d
from copy import deepcopy

from .base_model import BaseModel
from .custom_layers import conv2d, deconv2d
from .utils import cross_entropy
from .vgg16 import vgg16


def encoder(inputs, prefix, num_units, dropout_rate, trainable=True,
            is_training=False, reuse=True):
    """
    VGG16 image encoder with fusion of conv4_3 and conv5_3 features.

    Args:
        inputs: input tensor, in channel-last-format
        prefix: prefix of any variable name
        num_units: Number of feature units in the FCN.
        batch_normalization (bool): Whether or not to perform batch normalization.
        trainable (bool): If False, variables are not trainable.
        is_training (bool): Indicator whether batch_normalization should be in training
            (batch) or testing (continuous) mode.
        reuse (bool): If true, reuse existing variables of same name
            (attention with prefix). Will raise error if it cannot find such variables.
    Returns:
        dict of (intermediate) layer outputs
    """
    # These parameters are shared between many/all layers and therefore defined here
    # for convenience.
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': False, 'training': is_training,
              'trainable': trainable}

    l = vgg16(inputs, prefix, params)

    # dropout after pool3
    l['pool3_drop'] = dropout(l['pool3'], rate=dropout_rate, training=is_training,
                              name='{}_dropout'.format(prefix))
    # now we have to overwrite the following layers:
    params['reuse'] = True
    l['conv4_1'] = conv2d(l['pool3'], 512, [3, 3], name='{}_conv4_1'.format(prefix),
                          **params)
    l['conv4_2'] = conv2d(l['conv4_1'], 512, [3, 3], name='{}_conv4_2'.format(prefix),
                          **params)
    l['conv4_3'] = conv2d(l['conv4_2'], 512, [3, 3], name='{}_conv4_3'.format(prefix),
                          **params)
    l['pool4'] = max_pooling2d(l['conv4_3'], [2, 2], [2, 2],
                               name='{}_pool4'.format(prefix))
    l['conv5_1'] = conv2d(l['pool4'], 512, [3, 3],
                          name='{}_conv5_1'.format(prefix), **params)
    l['conv5_2'] = conv2d(l['conv5_1'], 512, [3, 3], name='{}_conv5_2'.format(prefix),
                          **params)
    l['conv5_3'] = conv2d(l['conv5_2'], 512, [3, 3], name='{}_conv5_3'.format(prefix),
                          **params)
    params['reuse'] = reuse

    # Use 1x1 convolutions on conv4_3 and conv5_3 to define features.
    score_conv4 = conv2d(l['conv4_3'], num_units, [1, 1],
                         name='{}_score_conv4'.format(prefix), **params)
    score_conv5 = conv2d(l['conv5_3'], num_units, [1, 1],
                         name='{}_score_conv5'.format(prefix), **params)
    # The deconvolution is always set static.
    params['trainable'] = False
    upscore_conv5 = deconv2d(score_conv5, num_units, [4, 4], strides=[2, 2],
                             name='{}_upscore_conv5'.format(prefix), **params)

    l['fused'] = tf.add_n([score_conv4, upscore_conv5],
                          name='{}_add_score'.format(prefix))

    # Return dictionary of all layers
    return l


def decoder(features, prefix, num_units, num_classes, trainable=True, is_training=False,
            reuse=True):
    """
    FCN feature decoder.

    Args:
        features: input tensor, in feature-last format
        prefix: prefix of any variable name
        num_units: Number of feature units in the FCN.
        num_classes: Number of output classes.
        dropout_rate: Dropout rate for dropout applied on input feature. Set to 0 to
            disable dropout.
        batch_normalization (bool): Whether or not to perform batch normalization.
        trainable (bool): If False, variables are not trainable.
        is_training (bool): Indicator whether batch_normalization should be in training
            (batch) or testing (continuous) mode.
        reuse (bool): If true, reuse existing variables of same name
            (attention with prefix). Will raise error if it cannot find such variables.
    Returns:
        dict of (intermediate) layer outputs
    """
    params = {
        'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
        'batch_normalization': False, 'training': is_training, 'trainable': trainable}
    # Upscore layers are never trainable
    upscore_params = deepcopy(params)
    upscore_params['trainable'] = False

    # Upsample the fused features to the output classification size
    features = deconv2d(features, num_units, [16, 16], strides=[8, 8],
                        name='{}_upscore'.format(prefix), **upscore_params)
    # no activation before the softmax
    params['activation'] = None
    score = conv2d(features, num_classes, [1, 1],
                   name='{}_score'.format(prefix), **params)
    return {'upscore': features, 'score': score}


class SimpleFCN(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers.

    Args:
        output_dir: if set, will output diagnostics and weights here
        learning_rate: learning rate of the trainer
        modality: name of the data modality, which has to be a valid key for the input
            dataset batch
        num_channels: channel-size of the input data (3 for RGB, 1 for Depth)

        Other Args see encoder and decoder
    """

    def __init__(self, prefix, data_description, modality, output_dir=None, **config):
        self.prefix = prefix
        self.modality = modality

        standard_config = {
            'train_encoder': True
        }
        standard_config.update(config)
        BaseModel.__init__(self, 'SimpleFCN', data_description, output_dir=output_dir,
                           **standard_config)

    def _build_graph(self, train_data, test_data):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Training network
        train_x = train_data[self.modality]
        # ground truth labels
        train_y = tf.to_float(train_data['labels'])

        encoder_layers = encoder(train_x, self.prefix, self.config['num_units'],
                                 self.config['dropout_rate'], is_training=True,
                                 reuse=False,
                                 trainable=self.config.get('train_encoder', True))
        decoder_layers = decoder(encoder_layers['fused'], self.prefix,
                                 self.config['num_units'], self.config['num_classes'],
                                 is_training=True, reuse=False)
        prob = tf.nn.log_softmax(decoder_layers['score'], name='prob')
        # The loss is given by the cross-entropy with the ground-truth
        self.loss = tf.div(tf.reduce_sum(cross_entropy(prob, train_y)),
                           tf.reduce_sum(train_y))

        # Network for testing / evaluation
        test_x = test_data[self.modality]
        encoder_layers = encoder(test_x, self.prefix, self.config['num_units'],
                                 tf.constant(0.0), is_training=False, reuse=True)
        decoder_layers = decoder(encoder_layers['fused'], self.prefix,
                                 self.config['num_units'], self.config['num_classes'],
                                 is_training=False, reuse=True)
        label = tf.nn.softmax(decoder_layers['score'], name='prob_normalized')
        self.prediction = tf.argmax(label, 3, name='label_2d')

        # Add summaries for some weights
        variable_names = [x.format(self.prefix)
                          for x in ['{}_score/kernel:0', '{}_score/bias:0']]
        for name in variable_names:
            var = next(v for v in tf.global_variables() if v.name == name)
            tf.summary.histogram(name, var)
