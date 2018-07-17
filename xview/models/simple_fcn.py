import tensorflow as tf
from tensorflow.python.layers.layers import dropout, max_pooling2d
from copy import deepcopy

from .base_model import BaseModel
from .custom_layers import conv2d, deconv2d
from .utils import cross_entropy


def encoder(inputs, prefix, num_units, dropout_rate, trainable=True, batchnorm=True,
            is_training=False, reuse=tf.AUTO_REUSE, dropout_layers=[]):
    """
    VGG16 image encoder with fusion of conv4_3 and conv5_3 features.

    Args:
        inputs: input tensor, in channel-last-format
        prefix: prefix of any variable name
        num_units: Number of feature units in the FCN.
        batchnorm (bool): Whether or not to perform batch normalization.
        trainable (bool): If False, variables are not trainable.
        is_training (bool): Indicator whether batch_normalization should be in training
            (batch) or testing (continuous) mode.
        reuse (bool): If true, reuse existing variables of same name
            (attention with prefix). Will raise error if it cannot find such variables.
        dropout_layers: a list of layers after which to apply dropout. Accepted possible
            values are 'pool3' and 'pool4'
    Returns:
        Dict of (intermediate) layer outputs. The encoding has key 'fused'.
    """
    # These parameters are shared between many/all layers and therefore defined here
    # for convenience.
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': batchnorm, 'training': is_training,
              'trainable': trainable}

    with tf.variable_scope(prefix, reuse=reuse):
        # dict for all the layers
        l = {}
        l['conv1_1'] = conv2d(inputs, 64, [3, 3], name='conv1_1', **params)
        l['conv1_2'] = conv2d(l['conv1_1'], 64, [3, 3], name='conv1_2', **params)
        l['pool1'] = max_pooling2d(l['conv1_2'], [2, 2], [2, 2], name='pool1')
        l['conv2_1'] = conv2d(l['pool1'], 128, [3, 3], name='conv2_1', **params)
        l['conv2_2'] = conv2d(l['conv2_1'], 128, [3, 3], name='conv2_2', **params)
        l['pool2'] = max_pooling2d(l['conv2_2'], [2, 2], [2, 2], name='pool2')
        l['conv3_1'] = conv2d(l['pool2'], 256, [3, 3], name='conv3_1', **params)
        l['conv3_2'] = conv2d(l['conv3_1'], 256, [3, 3], name='conv3_2', **params)
        l['conv3_3'] = conv2d(l['conv3_2'], 256, [3, 3], name='conv3_3', **params)
        l['pool3'] = max_pooling2d(l['conv3_3'], [2, 2], [2, 2], name='pool3')
        # dropout after pool3
        last_layer = l['pool3']
        if 'pool3' in dropout_layers:
            l['pool3_drop'] = dropout(l['pool3'], rate=dropout_rate, training=True,
                                      name='pool3_dropout')
            last_layer = l['pool3_drop']
        l['conv4_1'] = conv2d(last_layer, 512, [3, 3], name='conv4_1', **params)
        l['conv4_2'] = conv2d(l['conv4_1'], 512, [3, 3], name='conv4_2', **params)
        l['conv4_3'] = conv2d(l['conv4_2'], 512, [3, 3], name='conv4_3', **params)
        l['pool4'] = max_pooling2d(l['conv4_3'], [2, 2], [2, 2], name='pool4')
        # dropout after pool4
        last_layer = l['pool4']
        if 'pool3' in dropout_layers:
            l['pool4_drop'] = dropout(l['pool4'], rate=dropout_rate, training=True,
                                      name='pool4_dropout')
            last_layer = l['pool4_drop']
        l['conv5_1'] = conv2d(last_layer, 512, [3, 3], name='conv5_1', **params)
        l['conv5_2'] = conv2d(l['conv5_1'], 512, [3, 3], name='conv5_2', **params)
        l['conv5_3'] = conv2d(l['conv5_2'], 512, [3, 3], name='conv5_3', **params)
        # Use 1x1 convolutions on conv4_3 and conv5_3 to define features.
        # first, maybe apply dropout at these layers?
        conv4_3 = l['conv4_3']
        if 'conv4_3' in dropout_layers:
            conv4_3 = dropout(conv4_3, rate=dropout_rate, training=True,
                              name='conv4_3_dropout')
        score_conv4 = conv2d(conv4_3, num_units, [1, 1], name='%s_score_conv4' % prefix,
                             **params)
        conv5_3 = l['conv5_3']
        if 'conv5_3' in dropout_layers:
            conv5_3 = dropout(conv5_3, rate=dropout_rate, training=True,
                              name='conv5_3_dropout')
        score_conv5 = conv2d(conv5_3, num_units, [1, 1], name='%s_score_conv5' % prefix,
                             **params)
        # The deconvolution is always set static.
        params['trainable'] = False
        upscore_conv5 = deconv2d(score_conv5, num_units, [4, 4], strides=[2, 2],
                                 name='upscore_conv5', **params)

        l['fused'] = tf.add_n([score_conv4, upscore_conv5], name='add_score')
    # Return dictionary of all layers
    return l


def decoder(features, prefix, num_units, num_classes, trainable=True, is_training=False,
            reuse=tf.AUTO_REUSE, dropout_rate=None, batchnorm=True):
    """
    FCN feature decoder.

    Args:
        features: input tensor, in feature-last format
        prefix: prefix of any variable name
        num_units: Number of feature units in the FCN.
        num_classes: Number of output classes.
        dropout_rate: Dropout rate for dropout applied on input feature. Set to 0 to
            disable dropout.
        batchnorm (bool): Whether or not to perform batch normalization.
        trainable (bool): If False, variables are not trainable.
        is_training (bool): Indicator whether batch_normalization should be in training
            (batch) or testing (continuous) mode.
        reuse (bool): If true, reuse existing variables of same name
            (attention with prefix). Will raise error if it cannot find such variables.
        dropout_rate: If set, apply dropout on the decoder input with the given rate
    Returns:
        dict of (intermediate) layer outputs. The final per-class activations have key
            'score'.
    """
    # These parameters are shared between many/all layers and therefore defined here
    # for convenience.
    params = {
        'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
        'batch_normalization': batchnorm, 'training': is_training,
        'trainable': trainable}
    # Upscore layers are never trainable
    upscore_params = deepcopy(params)
    upscore_params['trainable'] = False

    with tf.variable_scope(prefix, reuse=reuse):
        if dropout_rate is not None:
            features = dropout(features, rate=dropout_rate, training=True,
                               name='features_dropout')

        # Upsample the fused features to the output classification size
        features = deconv2d(features, num_units, [16, 16], strides=[8, 8],
                            name='upscore', **upscore_params)
        # no activation before the softmax
        params['activation'] = None
        score = conv2d(features, num_classes, [1, 1], name='score', **params)
    return {'upscore': features, 'score': score}


def fcn(inputs, prefix, num_units, num_classes, trainable=True, is_training=False,
        reuse=tf.AUTO_REUSE, dropout_rate=0, dropout_layers=[], batchnorm=True):
    """
    Fully Convolutional Network with a VGG16 encoder and fusion of features from the
    4th and 5th pooling output before upconvolution. All upconvolutions are static
    bilinear interpolations and not trainable. Architecture was developed in DA-RNN.

    Args:
        inputs: input tensor, in channel-last-format
        prefix: prefix of any variable name
        num_units: Number of feature units in the FCN.
        num_classes: Number of output classes.
        batchnorm (bool): Whether or not to perform batch normalization.
        trainable (bool): If False, variables are not trainable.
        is_training (bool): Indicator whether batch_normalization should be in training
            (batch) or testing (continuous) mode.
        reuse (bool): If true, reuse existing variables of same name
            (attention with prefix). Will raise error if it cannot find such variables.
        dropout_layers (list): Layers after which to apply dropout. Accepted possible
            values are 'pool3', 'pool4' and 'fused'
        dropout_rate (float): The dropout probability applied to the specified layers.
    Returns:
        Dict of all (intermediate) layer outputs. The final per-class activations have key
            'score'.
    """
    layers = encoder(inputs, prefix, num_units, dropout_rate, is_training=is_training,
                     trainable=trainable, batchnorm=batchnorm,
                     dropout_layers=dropout_layers)
    layers.update(decoder(layers['fused'], prefix, num_units, num_classes,
                          is_training=is_training, batchnorm=batchnorm,
                          dropout_rate=(dropout_rate
                                        if 'features' in dropout_layers
                                        else None)))
    return layers


class SimpleFCN(BaseModel):
    """Model Implementation of FCN.

    Args:
        output_dir: if set, will output diagnostics and weights here
        learning_rate: learning rate of the trainer
        modality: name of the data modality, which has to be a valid key for the input
            dataset batch
        train_encoder (bool): If False, will keep the weights of the encoder static, may
            be useful for finetuning.
        num_units (int): Number of feature units in the FCN.
        prefix (string): prefix of any variable name
        batch_normalization (bool): Whether or not to use batch normalization.
    """

    def __init__(self, prefix, data_description, modality, output_dir=None, **config):
        self.prefix = prefix
        self.modality = modality

        standard_config = {
            'train_encoder': True,
            'dropout_rate': 0
        }
        standard_config.update(config)
        BaseModel.__init__(self, data_description, output_dir=output_dir,
                           **standard_config)

    def _build_graph(self, train_data, test_data):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Training network
        with tf.name_scope('training'):
            train_x = train_data[self.modality]
            # ground truth labels
            train_y = tf.to_float(train_data['labels'])
            layers = fcn(train_x, self.prefix, self.config['num_units'],
                         self.config['num_classes'], is_training=True,
                         batchnorm=self.config['batch_normalization'])
            prob = tf.nn.log_softmax(layers['score'], name='prob')
            # The loss is given by the cross-entropy with the ground-truth
            self.loss = cross_entropy(prob, train_y)
            self.train_y = train_y

        # Network for testing / evaluation
        with tf.name_scope('testing'):
            test_x = test_data[self.modality]
            layers = fcn(test_x, self.prefix, self.config['num_units'],
                         self.config['num_classes'], is_training=False,
                         batchnorm=self.config['batch_normalization'])
            prob = tf.nn.softmax(layers['score'], name='prob_normalized')
            self.prediction = tf.argmax(prob, 3, name='label_2d')
