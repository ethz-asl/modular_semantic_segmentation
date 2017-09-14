import tensorflow as tf
from tensorflow.python.layers.layers import dropout

from .base_model import BaseModel
from .custom_layers import conv2d, deconv2d, log_softmax, softmax
from .utils import cross_entropy
from .vgg16 import vgg16


def encoder(inputs, prefix, config, is_training=False, reuse=True):
    """VGG16 image encoder with fusion of conv4_3 and conv5_3 features."""
    # These parameters are shared between many/all layers and therefore defined here
    # for convenience.
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': config['batch_normalization'],
              'training': is_training, 'trainable': config['train_encoder']}

    vgg16_layers = vgg16(inputs, prefix, params)

    # Use 1x1 convolutions on conv4_3 and conv5_3 to define features.
    score_conv4 = conv2d(vgg16_layers['conv4_3'], config['num_units'], [1, 1],
                         name='{}_score_conv4'.format(prefix), **params)
    score_conv5 = conv2d(vgg16_layers['conv5_3'], config['num_units'], [1, 1],
                         name='{}_score_conv5'.format(prefix), **params)
    # The deconvolution is always set static.
    params['trainable'] = False
    upscore_conv5 = deconv2d(score_conv5, config['num_units'], [4, 4], strides=[2, 2],
                             name='{}_upscore_conv5'.format(prefix), **params)

    fused = tf.add_n([score_conv4, upscore_conv5], name='{}_add_score'.format(prefix))

    # Return dictionary of all layers
    ret = {'fused': fused}
    ret.update(vgg16_layers)
    return ret


def decoder(features, prefix, dropout_rate, config, is_training=False, reuse=True):
    """FCN feature decoder."""
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': config['batch_normalization'],
              'training': is_training}

    features = dropout(features, rate=dropout_rate,
                       name='{}_dropout'.format(prefix))
    # Upsample the fused features to the output classification size
    features = deconv2d(features, config['num_units'], [16, 16], strides=[8, 8],
                        name='{}_upscore'.format(prefix), trainable=False, **params)
    score = conv2d(features, config['num_classes'], [1, 1],
                   name='{}_score'.format(prefix), **params)
    return {'upscore': features, 'score': score}


class SimpleFCN(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers."""

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'train_encoder': True
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

        encoder_layers = encoder(train_x, self.config['modality'], self.config,
                                 is_training=True, reuse=False)
        decoder_layers = decoder(encoder_layers['fused'], self.config['modality'],
                                 train_dropout_rate, self.config,
                                 is_training=True, reuse=False)
        prob = log_softmax(decoder_layers['score'], self.config['num_classes'],
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

        encoder_layers = encoder(self.test_X, self.config['modality'], self.config)
        decoder_layers = decoder(encoder_layers['fused'], self.config['modality'],
                                 tf.constant(0.0), self.config)
        label = softmax(decoder_layers['score'], self.config['num_classes'],
                        name='prob_normalized')
        self.prediction = tf.argmax(label, 3, name='label_2d')

        # Add summaries for some weights
        variable_names = [x.format(self.config['modality'])
                          for x in ['{}_score/kernel:0', '{}_score/bias:0']]
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
