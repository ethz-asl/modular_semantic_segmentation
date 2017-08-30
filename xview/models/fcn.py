import tensorflow as tf
from tensorflow.python.layers.layers import max_pooling2d, dropout

from .base_model import BaseModel
from .custom_layers import conv2d, deconv2d, log_softmax, softmax
from .utils import cross_entropy


class FCN(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers.

    Documentation for some required methods can be found in BaseModel.
    """

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'batch_normalization': True
        }
        standard_config.update(config)
        BaseModel.__init__(self, 'FCN', output_dir=output_dir, **standard_config)

    def _build_graph(self):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Training network
        # First we define placeholders for the input into the input-queue.
        # This is not the direct input into the network, the enqueue-op has to be called
        # to evaluate any data that is fed here.
        # rgb channel
        self.train_X_rgb = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # depth channel
        self.train_X_d = tf.placeholder(tf.float32, shape=[None, None, None, 1])
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
        q = tf.FIFOQueue(5, [tf.float32, tf.float32, tf.float32, tf.float32])
        self.enqueue_op = q.enqueue([self.train_X_rgb, self.train_X_d, self.train_Y,
                                     self.train_dropout_rate])
        train_rgb, train_depth, training_labels, train_dropout_rate = q.dequeue()

        # This operation has to be called to close the input queue and free the space it
        # occupies in memory.
        self.close_queue_op = q.close(cancel_pending_enqueues=True)

        # The queue output does not have a defined shape, so we have to define it here to
        # be compatible with tf.layers.
        train_rgb.set_shape([None, None, None, 3])
        train_depth.set_shape([None, None, None, 1])

        train_score = self._fcn(train_rgb, train_depth, train_dropout_rate,
                                is_training=True, reuse=False)
        # Normalize probabilities to 1.
        train_prob = log_softmax(train_score, self.config['num_classes'], name='prob')
        # Loss is the cross-entropy with the ground truth
        self.loss = tf.div(tf.reduce_sum(cross_entropy(training_labels, train_prob)),
                           tf.reduce_sum(training_labels))

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        # rgb channel
        self.test_X_rgb = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # depth channel
        self.test_X_d = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        # dropout rate
        self.test_dropout_rate = tf.placeholder(tf.float32)
        test_score = self._fcn(self.test_X_rgb, self.test_X_d, self.test_dropout_rate)
        # For classification, we want to output the argmax of the probability vector.
        label = softmax(test_score, self.config['num_classes'], name='prob_normalized')
        self.prediction = tf.argmax(label, 3, name='label_2d')

    def _fcn(self, rgb, depth, dropout_rate, is_training=False, reuse=True):
        """Defines the core aarchitecture of the FCN network for RGB and Depth."""
        # These parameters are shared between many/all layers and therefore defined here
        # for convenience.
        params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
                  'batch_normalization': self.config['batch_normalization'],
                  'training': is_training}

        def vgg16(inputs, prefix):
            """VGG16 image encoder."""
            # common parameters
            net = conv2d(inputs, 64, [3, 3], name='{}_conv1_1'.format(prefix), **params)
            net = conv2d(net, 64, [3, 3], name='{}_conv1_2'.format(prefix), **params)
            net = max_pooling2d(net, [2, 2], [2, 2], name='{}_pool1'.format(prefix))
            net = conv2d(net, 128, [3, 3], name='{}_conv2_1'.format(prefix), **params)
            net = conv2d(net, 128, [3, 3], name='{}_conv2_2'.format(prefix), **params)
            net = max_pooling2d(net, [2, 2], [2, 2], name='{}_pool2'.format(prefix))
            net = conv2d(net, 256, [3, 3], name='{}_conv3_1'.format(prefix), **params)
            net = conv2d(net, 256, [3, 3], name='{}_conv3_2'.format(prefix), **params)
            net = conv2d(net, 256, [3, 3], name='{}_conv3_3'.format(prefix), **params)
            net = max_pooling2d(net, [2, 2], [2, 2], name='{}_pool3'.format(prefix))
            net = conv2d(net, 512, [3, 3], name='{}_conv4_1'.format(prefix), **params)
            net = conv2d(net, 512, [3, 3], name='{}_conv4_2'.format(prefix), **params)
            conv4 = conv2d(net, 512, [3, 3], name='{}_conv4_3'.format(prefix), **params)
            net = max_pooling2d(conv4, [2, 2], [2, 2], name='{}_pool4'.format(prefix))
            net = conv2d(net, 512, [3, 3], name='{}_conv5_1'.format(prefix), **params)
            net = conv2d(net, 512, [3, 3], name='{}_conv5_2'.format(prefix), **params)
            conv5 = conv2d(net, 512, [3, 3], name='{}_conv5_3'.format(prefix), **params)
            return conv5, conv4

        # Each input modality is encoded by a seperate VGG16 encoder.
        rgb_conv5, rgb_conv4 = vgg16(rgb, 'rgb')
        depth_conv5, depth_conv4 = vgg16(depth, 'depth')

        # At the output of stages conv4_3 and conv5_3, stack the 2 modalities together.
        conv4 = tf.concat([rgb_conv4, depth_conv4], 3, name='concat_conv4')
        conv4 = conv2d(conv4, self.config['num_units'], [1, 1], name='score_conv4',
                       **params)

        # Output of conv5_3 is upsampled to match the dimensions.
        conv5 = tf.concat([rgb_conv5, depth_conv5], 3, name='concat_conv5')
        conv5 = conv2d(conv5, self.config['num_units'], [1, 1], name='score_conv5',
                       **params)
        conv5 = deconv2d(conv5, self.config['num_units'], [4, 4], strides=[2, 2],
                         name='upscore_conv5', trainable=False, **params)

        fused = tf.add_n([conv4, conv5], name='add_score')
        fused = dropout(fused, rate=dropout_rate, name='dropout')

        # Upsample the fused features to the output classification size
        fused = deconv2d(fused, self.config['num_units'], [16, 16], strides=[8, 8],
                         name='upscore', trainable=False, **params)
        score = conv2d(fused, self.config['num_classes'], [1, 1], name='score',
                       **params)
        return score

    def _enqueue_batch(self, batch, sess):
        with self.graph.as_default():
            feed_dict = {self.train_X_rgb: batch['rgb'], self.train_X_d: batch['depth'],
                         self.train_Y: batch['labels'],
                         self.train_dropout_rate: self.config['dropout_rate']}
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def _evaluation_food(self, data):
        feed_dict = {self.test_X_rgb: data['rgb'], self.test_X_d: data['depth'],
                     self.test_dropout_rate: 0}
        return feed_dict
