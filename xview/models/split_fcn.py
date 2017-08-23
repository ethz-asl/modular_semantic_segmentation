import tensorflow as tf
from tensorflow.python.layers.layers import max_pooling2d, dropout

from .base_model import BaseModel
from .custom_layers import conv2d, deconv2d, log_softmax, softmax
from .utils import cross_entropy


class FCN(BaseModel):
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
        # First we define placeholders for the input into the input-queue.
        # This is not the direct input into the network, the enqueue-op has to be called
        # to evaluate any data that is fed here.
        # rgb channel
        self.X_rgb = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # depth channel
        self.X_d = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # ground truth labels
        self.Y_in = tf.placeholder(tf.float32, shape=[None, None, None,
                                                      self.config['num_classes']])
        # dropout keep probability (This is also an input as it will be different between
        # training and evaluation)
        self.dropout_rate = tf.placeholder(tf.float32)
        # training indicator, necessary for batch normalization, is a bool
        self.is_training = tf.placeholder(tf.bool)
        # An input queue is defined to load the data for several batches in advance and
        # keep the gpu as busy as we can.
        q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.float32, tf.float32, tf.bool])
        self.enqueue_op = q.enqueue([self.X_rgb, self.X_d, self.Y_in, self.dropout_rate,
                                     self.is_training])
        rgb, depth, training_labels, dropout_rate, is_training = q.dequeue()
        # The queue output does not have a defined shape, so we have to define it here to
        # be compatible with tf.layers.
        rgb.set_shape([None, None, None, 3])
        depth.set_shape([None, None, None, 3])
        # This operation has to be called to close the input queue and free the space it
        # occupies in memory.
        self.close_queue_op = q.close(cancel_pending_enqueues=True)

        # These parameters are shared between many/all layers and therefore defined here
        # for convenience.
        params = {'activation': tf.nn.relu, 'padding': 'same',
                  'batch_normalization': self.config['batch_normalization'],
                  'training': is_training}

        def vgg16(inputs, prefix):
            """VGG16 image encoder."""
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

        def decoder(features, prefix, reuse=True):
            decoder_params = params
            decoder_params['reuse'] = reuse

            features = dropout(features, rate=dropout_rate,
                               name='{}_dropout'.format(prefix))
            # Upsample the fused features to the output classification size
            features = deconv2d(features, self.config['num_units'], [16, 16],
                                strides=[8, 8], name='{}_upscore'.format(prefix),
                                trainable=False, **decoder_params)
            score = conv2d(features, self.config['num_classes'], [1, 1],
                           name='{}_score'.format(prefix), **decoder_params)
            return score

        # Each input modality is encoded by a seperate VGG16 encoder.
        rgb_features = vgg16(rgb, 'rgb')
        depth_features = vgg16(depth, 'depth')

        # For each modality, we generate training probabilities independently.
        rgb_score = decoder(rgb_features, 'rgb', reuse=False)
        rgb_prob = log_softmax(rgb_score, self.config['num_classes'], name='rgb_prob')
        depth_score = decoder(depth_features, 'depth', reuse=False)
        depth_prob = log_softmax(depth_score, self.config['num_classes'],
                                 name='depth_prob')
        # Based on these probabilites, we train both branches independently
        rgb_loss = tf.div(tf.reduce_sum(cross_entropy(training_labels, rgb_score)),
                          tf.reduce_sum(training_labels))
        tf.summary.scalar('rgb_loss', rgb_loss)
        self.rgb_trainer = tf.train.AdamOptimizer(self.config['learning_rate']).minimize(
            rgb_loss, global_step=self.global_step)
        depth_loss = tf.div(tf.reduce_sum(cross_entropy(training_labels, depth_score)),
                            tf.reduce_sum(training_labels))
        tf.summary.scalar('depth_loss', depth_loss)
        self.depth_trainer = tf.train.AdamOptimizer(self.config['learning_rate'])\
            .minimize(depth_loss, global_step=self.global_step)
        self.loss = tf.reduce_mean([rgb_loss, depth_loss])

        # For classification, we sample distributions with Dropout-Monte-Carlo and
        # fuse output according to variance
        rgb_samples = tf.stack([decoder(rgb_features, 'rgb')
                                for _ in range(self.config['num_samples'])],
                               axis=4)
        rgb_mean, rgb_var = tf.nn.moments(rgb_samples, [4])
        depth_samples = tf.stack([decoder(depth_features, 'depth')
                                  for _ in range(self.config['num_samples'])],
                                 axis=4)
        depth_mean, depth_var = tf.nn.moments(depth_samples, [4])

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
        rgb_available = tf.greater(tf.reduce_sum(rgb), 0.0)
        depth_available = tf.greater(tf.reduce_sum(depth), 0.0)
        self.prediction = tf.cond(rgb_available, lambda: tf.cond(depth_available,
                                                                 lambda: label,
                                                                 lambda: rgb_label),
                                  lambda: depth_label)

        # Now we expose the different used outputs as class properties.
        self.Y = training_labels

    def _enqueue_batch(self, batch, sess):
        with self.graph.as_default():
            feed_dict = {self.X_rgb: batch['rgb'], self.X_d: batch['depth'],
                         self.Y_in: batch['labels'],
                         self.dropout_rate: batch['dropout_rate'],
                         self.is_training: batch['is_training']}
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def _train_step(self, summaries):
        summaries, loss, _, _ = self.sess.run([summaries, self.loss, self.rgb_trainer,
                                               self.depth_trainer])
        return summaries, loss
