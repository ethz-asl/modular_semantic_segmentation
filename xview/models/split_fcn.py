import tensorflow as tf
from tensorflow.python.layers.layers import max_pooling2d, dropout

from .base_model import BaseModel
from .custom_layers import conv2d, deconv2d, log_softmax, softmax
from .utils import cross_entropy
from .vgg16 import vgg16


class SplitFCN(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers."""

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'num_samples': 20,
            'learning_rate': 0.01,
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

        # Now we handle both modalities independent
        def train_pipeline(inputs, prefix):
            features = self._encoder(inputs, prefix, is_training=True, reuse=False)
            score = self._decoder(features, prefix, train_dropout_rate,
                                  is_training=True, reuse=False)
            prob = log_softmax(score, self.config['num_classes'],
                               name='{}_prob'.format(prefix))
            # The loss is given by the cross-entropy with the ground-truth
            loss = tf.div(tf.reduce_sum(cross_entropy(training_labels, prob)),
                          tf.reduce_sum(training_labels))
            # Report the loss in summary
            tf.summary.scalar('{}_loss'.format(prefix), loss)
            # Define a trainer to minimize this loss
            trainer = tf.train.AdamOptimizer(self.config['learning_rate']).minimize(
                loss, global_step=self.global_step)
            return loss, trainer

        rgb_loss, self.rgb_trainer = train_pipeline(train_rgb, 'rgb')
        depth_loss, self.depth_trainer = train_pipeline(train_depth, 'depth')
        # for external measurements, simply report the mean loss
        self.loss = tf.reduce_mean([rgb_loss, depth_loss])

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        # rgb channel
        self.test_X_rgb = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # depth channel
        self.test_X_d = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        # dropout rate
        self.test_dropout_rate = tf.placeholder(tf.float32)

        def test_pipeline(inputs, prefix):
            features = self._encoder(inputs, prefix)
            # For classification, we sample distributions with Dropout-Monte-Carlo and
            # fuse output according to variance
            samples = tf.stack([self._decoder(features, 'rgb', self.test_dropout_rate)
                                for _ in range(self.config['num_samples'])], axis=4)
            mean, variance = tf.nn.moments(samples, [4])
            return mean, variance

        rgb_mean, rgb_var = test_pipeline(self.test_X_rgb, 'rgb')
        depth_mean, depth_var = test_pipeline(self.test_X_d, 'depth')

        fused_score = tf.multiply((rgb_var + depth_var),
                                  (rgb_mean / rgb_var + depth_mean / depth_var),
                                  name='variance_weighted_fusion')
        label = softmax(fused_score, self.config['num_classes'], name='prob_normalized')
        label = tf.argmax(label, 3, name='label_2d')

        # We also want to make predictions in the absense of one sensor modality.
        rgb_label = softmax(rgb_mean, self.config['num_classes'],
                            name='rgb_prob_normalized')
        rgb_label = tf.argmax(rgb_label, 3, name='rgb_label_2d')

        depth_label = softmax(depth_mean, self.config['num_classes'],
                              name='depth_prob_normalized')
        depth_label = tf.argmax(depth_label, 3, name='depth_label_2d')

        # Now we test which input data is available and based on this we produce the
        # corresponding prediction
        rgb_available = tf.greater(tf.reduce_sum(self.test_X_rgb), 0.0)
        depth_available = tf.greater(tf.reduce_sum(self.test_X_d), 0.0)
        self.prediction = tf.cond(rgb_available, lambda: tf.cond(depth_available,
                                                                 lambda: label,
                                                                 lambda: rgb_label),
                                  lambda: depth_label)

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
            feed_dict = {self.train_X_rgb: batch['rgb'], self.train_X_d: batch['depth'],
                         self.train_Y: batch['labels'],
                         self.train_dropout_rate: self.config['dropout_rate']}
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def _evaluation_food(self, data):
        feed_dict = {self.test_X_rgb: data['rgb'], self.test_X_d: data['depth'],
                     self.test_dropout_rate: self.config['sample_dropout_rate']}
        return feed_dict

    def _train_step(self, summaries):
        summaries, loss, _, _ = self.sess.run([summaries, self.loss, self.rgb_trainer,
                                               self.depth_trainer])
        return summaries, loss
