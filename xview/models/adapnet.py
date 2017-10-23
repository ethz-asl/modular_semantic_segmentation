import tensorflow as tf
from tensorflow.python.layers.layers import dropout, max_pooling2d

from .base_model import BaseModel
from .custom_layers import conv2d, deconv2d, log_softmax, softmax
from .utils import cross_entropy


# Definition of the different building blocks from the paper.


def block_a(inputs, intermed_filters, filters, strides, is_training, name, reuse,
            activation, shortcut_conv=False):
    # Common parameters of the convolutions
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': True, 'training': is_training}

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
    # Common parameters of the convolutions
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': True, 'training': is_training}
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


def adapnet(inputs, prefix, config, is_training=False, reuse=True):
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
        shortcut = conv2d(block_7, config['num_units'], 1, name='shortcut',
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
        deconv_1 = deconv2d(deconv_1, config['num_units'], 4, strides=2, activation=None,
                            reuse=reuse, name='first_deconvolution_upconv',
                            padding='same', batch_normalization=True,
                            training=is_training)

        merge = tf.add(deconv_1, shortcut)
        score = deconv2d(merge, config['num_classes'], 16, strides=8, activation=None,
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
    """FCN implementation following DA-RNN architecture and using tf.layers."""

    def __init__(self, prefix=None, output_dir=None, **config):
        standard_config = {
            'train_encoder': True
        }
        standard_config.update(config)

        if prefix is None:
            self.prefix = config['modality']
        else:
            self.prefix = prefix

        BaseModel.__init__(self, 'Adapnet', output_dir=output_dir, **standard_config)

    def _build_graph(self):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Training network
        # First we define placeholders for the input into the input-queue.
        # This is not the direct input into the network, the enqueue-op has to be called
        # to evaluate any data that is fed here.
        # rgb channel
        self.train_X = tf.placeholder(tf.float32, shape=[None, None, None,
                                                         self.config['num_channels']])
        # ground truth labels
        self.train_Y = tf.placeholder(tf.float32, shape=[None, None, None,
                                                         self.config['num_classes']])
        # dropout rate is also an input as it will be different between training and
        # evaluation
        self.train_dropout_rate = tf.placeholder(tf.float32)
        # An input queue is defined to load the data for several batches in advance and
        # keep the gpu as busy as we can.
        # IMPORTANT: The size of this queue can grow big very easily with growing
        # batchsize, therefore do not make the queue too long, otherwise we risk getting
        # killed by the OS
        q = tf.FIFOQueue(3, [tf.float32, tf.float32, tf.float32])
        self.enqueue_op = q.enqueue([self.train_X, self.train_Y,
                                     self.train_dropout_rate])
        train_x, training_labels, train_dropout_rate = q.dequeue()

        # This operation has to be called to close the input queue and free the space it
        # occupies in memory.
        self.close_queue_op = q.close(cancel_pending_enqueues=True)

        # The queue output does not have a defined shape, so we have to define it here to
        # be compatible with tf.layers.
        train_x.set_shape([None, None, None, self.config['num_channels']])

        adapnet_layers = adapnet(train_x, self.prefix, self.config, is_training=True,
                                 reuse=False)
        prob = log_softmax(adapnet_layers['score'], self.config['num_classes'],
                           name='prob')
        # The loss is given by the cross-entropy with the ground-truth
        self.loss = tf.div(tf.reduce_sum(cross_entropy(training_labels, prob)),
                           tf.reduce_sum(training_labels))

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        # rgb channel
        self.test_X = tf.placeholder(tf.float32, shape=[None, None, None,
                                                        self.config['num_channels']])

        adapnet_layers = adapnet(self.test_X, self.prefix, self.config,
                                 is_training=False, reuse=True)
        label = softmax(adapnet_layers['score'], self.config['num_classes'],
                        name='prob_normalized')
        self.prediction = tf.argmax(label, 3, name='label_2d')

        # Add summaries for some weights
        variable_names = []
        for name in variable_names:
            var = next(v for v in tf.global_variables() if v.name == name)
            tf.summary.histogram(name, var)

    def _enqueue_batch(self, batch, sess):
        with self.graph.as_default():
            feed_dict = {self.train_X: batch[self.config['modality']],
                         self.train_Y: batch['labels'],
                         self.train_dropout_rate: self.config['dropout_rate']}
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def _evaluation_food(self, data):
        feed_dict = {self.test_X: data[self.config['modality']]}
        return feed_dict
