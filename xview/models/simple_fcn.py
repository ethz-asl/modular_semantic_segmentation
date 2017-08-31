import tensorflow as tf
from tensorflow.python.layers.layers import max_pooling2d, dropout

from .base_model import BaseModel
from .custom_layers import conv2d, deconv2d, log_softmax, softmax
from .utils import cross_entropy
from .vgg16 import vgg16


class SimpleFCN(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers."""

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'num_samples': 20,
            'learning_rate': 0.01,
            'batch_normalization': True
        }
        standard_config.update(config)
        BaseModel.__init__(self, 'SimpleFCN', output_dir=output_dir, **standard_config)

    def _build_graph(self):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Training network
        # First we define placeholders for the input into the input-queue.
        # This is not the direct input into the network, the enqueue-op has to be called
        # to evaluate any data that is fed here.
        # rgb channel
        self.train_X_rgb = tf.placeholder(tf.float32, shape=[None, None, None, 3])
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
        q = tf.FIFOQueue(5, [tf.float32, tf.float32, tf.float32])
        self.enqueue_op = q.enqueue([self.train_X_rgb, self.train_Y,
                                     self.train_dropout_rate])
        train_rgb, training_labels, train_dropout_rate = q.dequeue()

        # This operation has to be called to close the input queue and free the space it
        # occupies in memory.
        self.close_queue_op = q.close(cancel_pending_enqueues=True)

        # The queue output does not have a defined shape, so we have to define it here to
        # be compatible with tf.layers.
        train_rgb.set_shape([None, None, None, 3])

        features = self._encoder(train_rgb, 'rgb', is_training=True, reuse=False)
        score = self._decoder(features, 'rgb', train_dropout_rate,
                              is_training=True, reuse=False)
        prob = log_softmax(score, self.config['num_classes'], name='prob')
        # The loss is given by the cross-entropy with the ground-truth
        self.loss = tf.div(tf.reduce_sum(cross_entropy(training_labels, prob)),
                           tf.reduce_sum(training_labels))

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        # rgb channel
        self.test_X_rgb = tf.placeholder(tf.float32, shape=[None, None, None, 3])

        features = self._encoder(self.test_X_rgb, 'rgb')
        score = self._decoder(features, 'rgb', tf.constant(0.0))
        label = softmax(score, self.config['num_classes'], name='prob_normalized')
        self.prediction = tf.argmax(label, 3, name='label_2d')

        # Add summaries for some weights
        variable_names = ['rgb_score/kernel:0', 'rgb_score/bias:0']
        for name in variable_names:
            var = next(v for v in tf.global_variables() if v.name == name)
            tf.summary.histogram(name, var)

    def _encoder(self, inputs, prefix, is_training=False, reuse=True):
        """VGG16 image encoder with fusion of conv4_3 and conv5_3 features."""
        # These parameters are shared between many/all layers and therefore defined here
        # for convenience.
        params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
                  'batch_normalization': self.config['batch_normalization'],
                  'training': is_training}

        conv4, conv5 = vgg16(inputs, prefix, params)

        # fuse features from conv4 and conv5 together
        conv4 = conv2d(conv4, self.config['num_units'], [1, 1],
                       name='{}_score_conv4'.format(prefix), **params)
        conv5 = conv2d(conv5, self.config['num_units'], [1, 1],
                       name='{}_score_conv5'.format(prefix), **params)
        conv5 = deconv2d(conv5, self.config['num_units'], [4, 4], strides=[2, 2],
                         name='{}_upscore_conv5'.format(prefix), trainable=False,
                         **params)

        fused = tf.add_n([conv4, conv5], name='{}_add_score'.format(prefix))
        return fused

    def _decoder(self, features, prefix, dropout_rate, is_training=False, reuse=True):
        params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
                  'batch_normalization': self.config['batch_normalization'],
                  'training': is_training}

        features = dropout(features, rate=dropout_rate,
                           name='{}_dropout'.format(prefix))
        # Upsample the fused features to the output classification size
        features = deconv2d(features, self.config['num_units'], [16, 16],
                            strides=[8, 8], name='{}_upscore'.format(prefix),
                            trainable=False, **params)
        score = conv2d(features, self.config['num_classes'], [1, 1],
                       name='{}_score'.format(prefix), **params)
        return score

    def _enqueue_batch(self, batch, sess):
        with self.graph.as_default():
            feed_dict = {self.train_X_rgb: batch['rgb'], self.train_Y: batch['labels'],
                         self.train_dropout_rate: self.config['dropout_rate']}
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def _evaluation_food(self, data):
        feed_dict = {self.test_X_rgb: data['rgb']}
        return feed_dict
