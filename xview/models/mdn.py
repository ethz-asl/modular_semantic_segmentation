import tensorflow as tf

from .adapnet import adapnet
from .toymodels import min_dense, min_cnn
from .simple_fcn import fcn
from .uncertainty_model import UncertaintyModel
from .custom_layers import softmax, entropy
from .utils import cross_entropy


class MixtureDensityNetwork(UncertaintyModel):
    def __init__(self, prefix, data_description, modality, **config):
        self.prefix = prefix
        self.modality = modality

        standard_config = {
            'expert_model': 'fcn',
            'num_mixtures': 4,
        }
        standard_config.update(config)
        UncertaintyModel.__init__(self, data_description, **standard_config)

    def _build_graph(self, train_data, test_data):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        output_classes = self.config['num_mixtures'] * (self.config['num_classes'] + 1)

        # Training network
        with tf.name_scope('training'):
            train_x = train_data[self.modality]
            # ground truth labels
            train_y = tf.to_float(train_data['labels'])
            if self.config['expert_model'] == 'fcn':
                layers = fcn(train_x, self.prefix, self.config['num_units'],
                             output_classes, is_training=True,
                             batchnorm=self.config['batch_normalization'])
            elif self.config['expert_model'] == 'adapnet':
                layers = adapnet(train_x, self.prefix, self.config['num_units'],
                                 output_classes, is_training=True)
            elif self.config['expert_model'] == 'min_dense':
                layers = min_dense(train_x, self.config['num_units'], output_classes)
            elif self.config['expert_model'] == 'min_cnn':
                layers = min_cnn(train_x, self.config['num_units'], output_classes,
                                 is_training=True,
                                 batchnorm=self.config['batch_normalization'])
            else:
                raise UserWarning('expert model not found')

            scores, weight_scores = tf.split(
                layers['score'],
                [self.config['num_mixtures'] * self.config['num_classes'],
                 self.config['num_mixtures']], axis=-1)
            # reshape the scores into <num_mixtures> scores of size <num_classes>
            target_shape = [tf.shape(train_x)[i] for i in range(len(train_x.shape))]
            target_shape[-1] = self.config['num_mixtures']
            target_shape.append(self.config['num_classes'])
            probs = tf.nn.softmax(tf.reshape(scores, target_shape))
            weights = tf.nn.softmax(weight_scores)
            prob = tf.reduce_sum(tf.expand_dims(weights, axis=-1) * probs, axis=-2)
            mean_variance = tf.reduce_mean(
                tf.reduce_sum(tf.expand_dims(weights, axis=-1) * probs *
                              (probs - tf.expand_dims(prob, axis=-2)),
                              axis=-2))
            # The loss is given by the cross-entropy with the ground-truth
            self.loss = cross_entropy(tf.log(prob), train_y)
                # - 0.0001 * mean_variance
                #+ 0.2 * tf.reduce_mean(entropy(probs))
                #+ 0.001 * tf.reduce_mean(tf.reduce_sum(entropy(probs) * weights, axis=-1))
                #- tf.reduce_mean(0.01 * entropy(weights, axis=-2))
            self.train_y = train_y
            tf.summary.scalar('weight_entropy', tf.reduce_mean(entropy(weights)))
            tf.summary.scalar('prob_entropy', tf.reduce_mean(entropy(prob)))
            tf.summary.scalar('mix_entropy', tf.reduce_mean(
                tf.reduce_sum(entropy(probs) * weights, axis=-1)))
            tf.summary.scalar('mean_variance', mean_variance)

        # Network for testing / evaluation
        with tf.name_scope('testing'):
            test_x = test_data[self.modality]
            if self.config['expert_model'] == 'fcn':
                layers = fcn(test_x, self.prefix, self.config['num_units'],
                             output_classes, is_training=False,
                             batchnorm=self.config['batch_normalization'])
            elif self.config['expert_model'] == 'adapnet':
                layers = adapnet(test_x, self.prefix, self.config['num_units'],
                                 output_classes, is_training=False)
            elif self.config['expert_model'] == 'min_dense':
                layers = min_dense(test_x, self.config['num_units'], output_classes)
            elif self.config['expert_model'] == 'min_cnn':
                layers = min_cnn(test_x, self.config['num_units'], output_classes,
                                 is_training=True,
                                 batchnorm=self.config['batch_normalization'])
            else:
                raise UserWarning('expert model not found')
            scores, weight_scores = tf.split(
                layers['score'],
                [self.config['num_mixtures'] * self.config['num_classes'],
                 self.config['num_mixtures']], axis=-1)
            # reshape the scores into <num_mixtures> scores of size <num_classes>
            target_shape = [tf.shape(test_x)[i] for i in range(len(test_x.shape))]
            target_shape[-1] = self.config['num_mixtures']
            target_shape.append(self.config['num_classes'])
            probs = tf.nn.softmax(tf.reshape(scores, target_shape))
            weights = tf.nn.softmax(weight_scores)
            self.prob = tf.reduce_sum(probs * tf.expand_dims(weights, axis=-1), axis=-2)
            self.prediction = tf.argmax(self.prob, -1, name='label_2d')
            self.entropy = entropy(self.prob)
            self.mix_entropy = tf.reduce_sum(entropy(probs) * weights, axis=-1)
            self.weight_entropy = entropy(weights)
            self.mean_variance = tf.reduce_mean(
                tf.reduce_sum(tf.expand_dims(weights, axis=-1) * (1 - probs) * probs,
                              axis=-2),
                axis=-1)
            self.var_over_mix = tf.reduce_mean(
                tf.reduce_sum(tf.expand_dims(weights, axis=-1) * probs *
                              (probs - tf.expand_dims(self.prob, axis=-2)),
                              axis=-2),
                axis=-1)
            self.weighted_mean_variance = tf.reduce_sum(
                self.prob *
                tf.reduce_sum(tf.expand_dims(weights, axis=-1) * (1 - probs) * probs,
                              axis=-2),
                axis=-1)
            self.weighted_var_over_mix = tf.reduce_sum(
                self.prob *
                tf.reduce_sum(tf.expand_dims(weights, axis=-1) * probs *
                              (probs - tf.expand_dims(self.prob, axis=-2)),
                              axis=-2),
                axis=-1)
