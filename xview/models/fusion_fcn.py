import tensorflow as tf
from copy import deepcopy

from .custom_layers import conv2d, deconv2d, log_softmax
from .utils import cross_entropy
from .base_model import BaseModel
from .simple_fcn import vgg16, decoder


def fusion_fcn(inputs, prefixes, num_units, num_classes, trainable=True,
               is_training=False, reuse=False):
    # These parameters are shared between many/all layers and therefore defined here
    # for convenience.
    params = {'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
              'batch_normalization': False, 'training': is_training,
              'trainable': trainable}
    deconv_params = deepcopy(params)
    deconv_params['trainable'] = False

    layers = {}
    for modality, prefix in prefixes.items():
        layers[modality] = vgg16(inputs[modality], prefix, params)
    # concat modalities
    layers['concat_conv4'] = tf.concat([layers[m]['conv4_3'] for m in prefixes], axis=3)
    layers['concat_conv5'] = tf.concat([layers[m]['conv5_3'] for m in prefixes], axis=3)
    # Use 1x1 convolutions to define features.
    layers['score_conv4'] = conv2d(layers['concat_conv4'], num_units, [1, 1],
                                   name='fused_score_conv4', **params)
    layers['score_conv5'] = conv2d(layers['concat_conv5'], num_units, [1, 1],
                                   name='fused_score_conv5', **params)
    layers['upscore_conv5'] = deconv2d(layers['score_conv5'], num_units, [4, 4],
                                       strides=[2, 2], name='fused_upscore_conv5',
                                       **deconv_params)

    layers['features'] = tf.add_n([layers['score_conv4'], layers['upscore_conv5']],
                                  name='fused_add_score')
    layers.update(decoder(layers['features'], 'fused', num_units, num_classes,
                          trainable=trainable, is_training=is_training, reuse=reuse))
    return layers


class FusionFCN(BaseModel):

    def __init__(self, prefixes, num_channels, num_units, num_classes, trainer='rmsprop',
                 learning_rate=0.0001, output_dir=None):
        self.modalities = list(prefixes.keys())

        BaseModel.__init__(self, 'FusionFCN', output_dir=output_dir,
                           num_channels=num_channels, num_units=num_units,
                           num_classes=num_classes, learning_rate=learning_rate,
                           trainer=trainer, prefixes=prefixes)

    def _build_graph(self):
        # --- TRAINING ---
        self.train_placeholders = {}
        for modality in self.modalities:
            self.train_placeholders[modality] = tf.placeholder(
                tf.float32, shape=[None, None, None,
                                   self.config['num_channels'][modality]])
        self.train_Y = tf.placeholder(tf.float32, shape=[None, None, None,
                                                         self.config['num_classes']])
        # An input queue is defined to load the data for several batches in advance and
        # keep the gpu as busy as we can.
        # IMPORTANT: The size of this queue can grow big very easily with growing
        # batchsize, therefore do not make the queue too long, otherwise we risk getting
        # killed by the OS
        q = tf.FIFOQueue(10, [tf.float32 for _ in range(len(self.modalities) + 1)])
        queue = [self.train_Y]
        queue.extend([self.train_placeholders[modality] for modality in self.modalities])
        self.enqueue_op = q.enqueue(queue)
        minibatches = q.dequeue()
        training_labels = minibatches[0]
        train_data = {self.modalities[i]: minibatches[i+1]
                      for i in range(len(self.modalities))}
        # The queue output does not have a defined shape, so we have to define it here to
        # be compatible with tf.layers.
        for modality in self.modalities:
            train_data[modality].set_shape([None, None, None,
                                            self.config['num_channels'][modality]])

        # This operation has to be called to close the input queue and free the space it
        # occupies in memory.
        self.close_queue_op = q.close(cancel_pending_enqueues=True)
        self.queue_is_empty_op = tf.equal(q.size(), 0)
        # To support tensorflow 1.2, we have to set this flag manually.
        self.queue_is_closed = False

        score = fusion_fcn(train_data, self.config['prefixes'], self.config['num_units'],
                           self.config['num_classes'], is_training=True,
                           reuse=False)['score']
        prob = log_softmax(score, self.config['num_classes'], name='prob')
        # The loss is given by the cross-entropy with the ground-truth
        self.loss = tf.div(tf.reduce_sum(cross_entropy(training_labels, prob)),
                           tf.reduce_sum(training_labels))

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        self.test_placeholders = {}
        for modality, channels in self.config['num_channels'].items():
            self.test_placeholders[modality] = tf.placeholder(
                tf.float32, shape=[None, None, None, channels])

        score = fusion_fcn(self.test_placeholders, self.config['prefixes'],
                           self.config['num_units'], self.config['num_classes'],
                           is_training=False, reuse=True)['score']
        label = tf.nn.softmax(score, name='prob_normalized')
        self.prediction = tf.argmax(label, 3, name='label_2d')

    def _evaluation_food(self, data):
        feed_dict = {self.test_placeholders[modality]: data[modality]
                     for modality in self.modalities}
        return feed_dict

    def _enqueue_batch(self, batch, sess):
        with self.graph.as_default():
            feed_dict = {self.train_Y: batch['labels']}
            for modality in self.modalities:
                feed_dict[self.train_placeholders[modality]] = batch[modality]
            sess.run(self.enqueue_op, feed_dict=feed_dict)
