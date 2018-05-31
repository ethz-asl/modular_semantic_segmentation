import tensorflow as tf

from .custom_layers import conv2d


def min_dense(inputs, num_units, num_classes, reuse=tf.AUTO_REUSE):
    params = {'activation': tf.nn.relu, 'reuse': reuse}

    l = {}
    l['hidden1'] = tf.layers.dense(inputs, num_units, **params)
    l['hidden2'] = tf.layers.dense(l['hidden1'], num_units, **params)
    l['score'] = tf.layers.dense(l['hidden2'], num_classes, **params)
    return l


def min_cnn(inputs, num_units, num_classes, reuse=tf.AUTO_REUSE, batchnorm=True,
            is_training=False):
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': batchnorm, 'training': is_training}

    l = {}
    l['1_1'] = conv2d(inputs, 4, [3, 3], name='conv1_1', **params)
    l['1_2'] = conv2d(l['1_1'], 4, [3, 3], name='conv1_2', **params)
    l['pool1'] = tf.layers.max_pooling2d(l['1_2'], [2, 2], [2, 2], name='pool1')
    l['2_1'] = conv2d(l['pool1'], 8, [3, 3], name='conv2_1', **params)
    l['2_2'] = conv2d(l['2_1'], num_units, [3, 3], name='conv2_2', **params)
    l['score'] = tf.layers.dense(l['2_2'], num_classes, activation=params['activation'],
                                 reuse=params['reuse'])
    return l
