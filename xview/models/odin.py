import tensorflow as tf

from .simple_fcn import fcn
from .uncertainty_model import UncertaintyModel
from .custom_layers import softmax, entropy
from .utils import cross_entropy


class ODIN(UncertaintyModel):
    def __init__(self, prefix, data_description, modality, **config):
        self.prefix = prefix
        self.modality = modality

        standard_config = {
            'temperature_scaling': 1
        }
        standard_config.update(config)
        UncertaintyModel.__init__(self, data_description, **standard_config)

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
            self.prob = softmax(layers['score'],
                                temperature=self.config['temperature_scaling'])
            self.prediction = tf.argmax(self.prob, 3, name='label_2d')
            self.entropy = entropy(self.prob)
            self.max_prob = 1 - tf.reduce_max(self.prob, 3, keepdims=True)

        # Add summaries for some weights
