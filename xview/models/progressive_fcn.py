import tensorflow as tf
from tensorflow.python.layers.layers import dropout

from .base_model import BaseModel
from .custom_layers import conv2d, deconv2d, log_softmax, softmax, adap_conv
from .utils import cross_entropy
from .vgg16 import progressive_vgg16
from .simple_fcn import encoder, decoder

from copy import deepcopy


def column(inputs, prefix, config, dropout_rate, other_columns=False, trainable=False,
           reuse=True):
    """
    FCN Column to be used in a progressive network setup.

    Args:
        inputs: The input tensor to the FCN
        prefix: Name prefix applied to all variables
        config: network config dict
        other_columns: a list of dicts for the output tensors of other columns of this
            type, dict should be exactly the return dict from this method
        reuse (bool): If true, reuse existing variables of same name
            (attention with prefix). Will raise error if it cannot find such variables.
    Returns:
        dict of (intermediate) layer outputs
    """
    # These parameters are shared between many/all layers and therefore defined here
    # for convenience.
    upscore_params = params = {
        'activation': tf.nn.relu, 'padding': 'same', 'reuse': reuse,
        'batch_normalization': False, 'training': False}
    # Deconvolutions are not trainable
    upscore_params['trainable'] = False
    params['trainable'] = trainable
    # extra adapter parameters
    adapter_params = {'trainable': trainable,
                      'extra_convolution': config['extra_adapter_convolution'],
                      'initial_scales': config['initial_adapter_scales']}

    # fusion of params and adapter-params
    all_adapter_params = deepcopy(params)
    all_adapter_params.update(adapter_params)

    # We collect all layer outputs in this dict
    layers = {}

    # In case that we already have other columns, create this column as a progressive
    # network on basis of the other columns. If not, define standalone
    if not other_columns:
        layers.update(encoder(inputs, prefix, config, reuse=reuse))
        layers.update(decoder(layers['fused'], prefix, tf.constant(0.0), config,
                              reuse=reuse))
    else:
        # Define the progressive version of FCN.
        layers.update(progressive_vgg16(inputs, other_columns, prefix, params,
                                        adapter_params))

        # Use 1x1 convolutions on conv4_3 and conv5_3 to define features.
        score_conv4 = conv2d(layers['conv4_3'], config['num_units'], [1, 1],
                             name='{}_score_conv4'.format(prefix), **params)
        score_conv5 = conv2d(layers['conv5_3'], config['num_units'], [1, 1],
                             name='{}_score_conv5'.format(prefix), **params)
        upscore_conv5 = deconv2d(score_conv5, config['num_units'], [4, 4],
                                 strides=[2, 2], name='{}_upscore_conv5'.format(prefix),
                                 **upscore_params)
        fused = tf.add_n([score_conv4, upscore_conv5],
                         name='{}_add_score'.format(prefix))
        layers['fused'] = fused

        # Now upsample the features and produce class scores.
        features = dropout(layers['fused'], rate=dropout_rate,
                           name='{}_dropout'.format(prefix))
        upscore = deconv2d(features, config['num_units'], [16, 16], strides=[8, 8],
                           name='{}_upscore'.format(prefix), **upscore_params)
        layers['upscore'] = upscore
        score = adap_conv(layers['upscore'], other_columns['upscore'],
                          config['num_classes'], [1, 1], name='{}_score'.format(prefix),
                          **all_adapter_params)
        layers['score'] = score
        combined_score = adap_conv(layers['score'], other_columns['score'],
                                   config['num_classes'], [1, 1],
                                   name='{}_combine_score'.format(prefix),
                                   **all_adapter_params)
        layers['combined_score'] = combined_score
    return layers


class ProgressiveFCN(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers.

    Args:
        prefix: prefix of all defined variables and operations for the newly trained
            column
        existing_columns: list of prefixes for already existing columns, weights need to
            be imported
        output_dir: If specified, the diagnostics will be saved in this directory.

        [Network specific config:]
        num_channels: number of channels in the input modality (eg 3 for RGB)
        num_classes: number of classes the classifier output should produce
        num_units: number of feature channels in the encoder output
        modality: name of the input modality picked from the data dict (eg 'depth')
        batch_normalization: Bool, whether to apply batch_normalization to convolutional
            layers
        dropout_rate: dropout-rate applied during training on the feature layer
    """

    def __init__(self, prefix, existing_columns, output_dir=None, **config):
        self.prefix = prefix
        self.existing_columns = existing_columns

        standard_config = {
            'train_encoder': True,
            'batch_normalization': False
        }
        standard_config.update(config)
        BaseModel.__init__(self, 'ProgressiveFCN', output_dir=output_dir,
                           **standard_config)

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
        q = tf.FIFOQueue(2, [tf.float32, tf.float32, tf.float32])
        self.enqueue_op = q.enqueue([self.train_X, self.train_Y,
                                     self.train_dropout_rate])
        train_x, training_labels, train_dropout_rate = q.dequeue()

        # This operation has to be called to close the input queue and free the space it
        # occupies in memory.
        self.close_queue_op = q.close(cancel_pending_enqueues=True)

        # The queue output does not have a defined shape, so we have to define it here to
        # be compatible with tf.layers.
        train_x.set_shape([None, None, None, self.config['num_channels']])

        # Set up exisiting columns
        train_columns = {}
        for prefix in self.existing_columns:
            new_column = column(train_x, prefix, self.config, train_dropout_rate,
                                other_columns=train_columns, reuse=False)
            for key, output in new_column.items():
                if key not in train_columns:
                    # This is the first column and we have to create keys.
                    train_columns[key] = [output]
                else:
                    train_columns[key].append(output)

        # Set up our new, trainable column
        train_layers = column(train_x, self.prefix, self.config, train_dropout_rate,
                              other_columns=train_columns, trainable=True, reuse=False)
        prob = log_softmax(train_layers['combined_score'], self.config['num_classes'],
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

        # Set up exisiing columns
        test_columns = {}
        for prefix in self.existing_columns:
            new_column = column(self.test_X, prefix, self.config, tf.constant(0.0),
                                other_columns=test_columns, reuse=True)
            for key, output in new_column.items():
                if key not in test_columns:
                    # This is the first column and we have to create keys.
                    test_columns[key] = [output]
                else:
                    test_columns[key].append(output)
        # Set up evaluation of this column
        test_layers = column(self.test_X, self.prefix, self.config, tf.constant(0.0),
                             other_columns=test_columns, reuse=True)
        label = softmax(test_layers['combined_score'], self.config['num_classes'],
                        name='prob_normalized')
        self.prediction = tf.argmax(label, 3, name='label_2d')

        # Add summaries for some weights
        variable_names = [x.format(self.prefix)
                          for x in ['{}_score/combination/kernel:0',
                                    '{}_score/combination/bias:0']]
        for name in variable_names:
            var = next(v for v in tf.global_variables() if v.name == name)
            tf.summary.histogram(name, var)

        # add summary for the adapters
        for var in tf.global_variables():
            if 'scale' in var.name:
                tf.summary.histogram(var.name, var)

    def _enqueue_batch(self, batch, sess):
        with self.graph.as_default():
            feed_dict = {self.train_X: batch[self.config['modality']],
                         self.train_Y: batch['labels'],
                         self.train_dropout_rate: self.config['dropout_rate']}
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def _evaluation_food(self, data):
        feed_dict = {self.test_X: data[self.config['modality']]}
        return feed_dict
