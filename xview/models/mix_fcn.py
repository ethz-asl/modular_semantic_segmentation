import tensorflow as tf
import numpy as np
from experiments.utils import ExperimentData
from os import path
from types import GeneratorType

from .dirichletEstimation import findDirichletPriors
from .base_model import BaseModel
from xview.models.simple_fcn import encoder, decoder


def bayes_fusion(probs, conditionals, prior):
    num_classes = probs[0].shape[3]

    # We will collect all posteriors in this list
    log_likelihoods = []
    for i_expert in range(len(conditionals)):
        # compute p(expert output | groudn truth class x)
        log_likelihood = tf.stack([conditionals[i_expert][c].log_prob(probs[i_expert])
                                   for c in range(num_classes)], axis=3)
        log_likelihoods.append(log_likelihood)

    return tf.reduce_sum(tf.stack(log_likelihoods, axis=0), axis=0) + tf.log(prior)


class MixFCN(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers."""

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'learning_rate': 0.0,
            'batch_normalization': False
        }
        standard_config.update(config)

        # If specified, load mesaurements of the experts.
        if 'measurement_exp' in config:
            measurements = np.load(ExperimentData(config["measurement_exp"])
                                   .get_artifact("counts.npz"))
            self.pseudocounts = {'rgb': measurements['rgb'],
                                 'depth': measurements['depth']}
            self.class_counts = measurements['class_counts']
        else:
            print('WARNING: Could not yet import measurements, you need to fit this '
                  'model first.')

        BaseModel.__init__(self, 'MixFCN', output_dir=output_dir,
                           supports_training=False, **config)

    def _build_graph(self):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        # rgb channel
        self.test_X_rgb = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # depth channel
        self.test_X_d = tf.placeholder(tf.float32, shape=[None, None, None, 1])

        def test_pipeline(inputs, prefix):
            # Now we get the network output of the FCN expert.
            outputs = {}
            outputs.update(encoder(inputs, prefix, self.config, reuse=False))
            outputs.update(decoder(outputs['fused'], prefix, 0.0, self.config,
                                   reuse=False))
            prob = tf.nn.softmax(outputs['score'])
            return prob

        self.rgb_prob = tf.cast(test_pipeline(self.test_X_rgb, 'rgb'), tf.float32)
        self.depth_prob = tf.cast(test_pipeline(self.test_X_d, 'depth'), tf.float32)

        rgb_label = tf.argmax(self.rgb_prob, 3, name='rgb_label_2d')
        depth_label = tf.argmax(self.depth_prob, 3, name='depth_label_2d')

        # The following part can only be build if measurements are already present.
        if hasattr(self, 'dirichlet_params'):
            # Create all the Dirichlet distributions conditional on ground-truth class
            rgb_dirichlets = {}
            depth_dirichlets = {}
            for c in range(self.config['num_classes']):
                rgb_dirichlets[c] = tf.contrib.distributions.Dirichlet(
                    self.dirichlet_params['rgb'][:, c].astype('float32'))
                depth_dirichlets[c] = tf.contrib.distributions.Dirichlet(
                    self.dirichlet_params['depth'][:, c].astype('float32'))

            # Set the Prior of the classes
            uniform_prior = 1.0 / 14
            data_prior = (self.class_counts / self.class_counts.sum()).astype('float32')
            if self.config['class_prior'] == 'uniform':
                # set a uniform prior for all classes
                prior = uniform_prior
            elif self.config['class_prior'] == 'data':
                prior = data_prior
            else:
                # The class_prior parameter is now considered a weight for the mixture
                # between both priors.
                weight = float(self.config['class_prior'])
                prior = weight * uniform_prior + (1 - weight) * data_prior
                prior = prior / prior.sum()

            fused_score = bayes_fusion([self.rgb_prob, self.depth_prob],
                                       [rgb_dirichlets, depth_dirichlets],
                                       prior)
            label = tf.argmax(fused_score, 3, name='label_2d')

            # Now we test which input data is available and based on this we produce the
            # corresponding prediction
            rgb_available = tf.greater(tf.reduce_sum(self.test_X_rgb), 0.0)
            depth_available = tf.greater(tf.reduce_sum(self.test_X_d), 0.0)

            self.prediction = tf.cond(rgb_available, lambda: tf.cond(depth_available,
                                                                     lambda: label,
                                                                     lambda: rgb_label),
                                      lambda: depth_label)
            self.prediction = label
        else:
            self.prediction = rgb_label
        # To understand what"s going on under the hood, we expose a lot of intermediate
        # results for evaluation
        self.rgb_branch = {'label': rgb_label}
        self.depth_branch = {'label': depth_label}

    def _enqueue_batch(self, batch, sess):
        # This model does not support training
        pass

    def fit(self, data):
        """Measure the encoder outputs against the given groundtruth for the given data.

        Args:
            data: As usual, an instance of DataWrapper
        """
        num_classes = self.config['num_classes']

        def get_sufficient_statistic(probs, labels, eps=1e-10):
            """To infer the hidden Dirichlet distribution, sum all prob vectors belonging
            to a given ground-truth label togehter to sufficient statistics."""
            sufficient_statistic = np.zeros((num_classes, num_classes))
            for c in range(num_classes):
                this_class = np.where(
                    np.stack([labels for _ in range(num_classes)], axis=-1) == c, probs,
                    np.ones_like(probs))
                # Sum the probability putput over batch and x, y coords.
                sufficient_statistic[:, c] += np.log(eps + this_class).sum((0, 1, 2))
            return sufficient_statistic

        def get_classcounts(labels):
            """Count the number of occurences per class."""
            counts = np.zeros(num_classes)
            for c in range(num_classes):
                counts[c] = (labels == c).sum()
            return counts

        with self.graph.as_default():
            # store all measurements in these matrices
            rgb_measurements = np.zeros((num_classes, num_classes))
            depth_measurements = np.zeros((num_classes, num_classes))
            class_counts = np.zeros(num_classes)

            if isinstance(data, GeneratorType):
                for batch in data:
                    rgb_prob, depth_prob = self.sess.run(
                        [self.rgb_prob, self.depth_prob],
                        feed_dict=self._evaluation_food(batch))
                    rgb_measurements += get_sufficient_statistic(rgb_prob,
                                                                 batch['labels'])
                    depth_measurements += get_sufficient_statistic(depth_prob,
                                                                   batch['labels'])
                    class_counts += get_classcounts(batch['labels'])
            else:
                rgb_prob, depth_prob = self.sess.run(
                    [self.rgb_prob, self.depth_prob],
                    feed_dict=self._evaluation_food(data))
                rgb_measurements += get_sufficient_statistic(rgb_prob, data['labels'])
                depth_measurements += get_sufficient_statistic(depth_prob,
                                                               data['labels'])
                class_counts += get_classcounts(data['labels'])

        # Now, given the sufficient statistic, run Expectation-Maximization to get the
        # Dirichlet parameters
        def dirichlet_em(measurements):
            """Find dirichlet parameters for all class-conditional dirichlets in
            measurements."""
            params = np.ones((num_classes, num_classes))

            for c in range(num_classes):
                params[:, c] = findDirichletPriors(measurements[:, c] / class_counts[c],
                                                   np.ones(num_classes))
            return params

        rgb_dirichlet_params = dirichlet_em(rgb_measurements)
        depth_dirichlet_params = dirichlet_em(depth_measurements)

        # Store the results
        self.dirichlet_params = {'rgb': rgb_dirichlet_params,
                                 'depth': depth_dirichlet_params}
        self.class_counts = class_counts
        if self.output_dir is not None:
            np.savez(path.join(self.output_dir, 'counts.npz'), class_counts=class_counts,
                     rgb=rgb_dirichlet_params, depth=depth_dirichlet_params)

        # Rebuild the graph with the new measurements:
        self._initialize_graph()

        print("INFO: MixFCN fitted to data")

    def _evaluation_food(self, data):
        feed_dict = {self.test_X_rgb: data['rgb'], self.test_X_d: data['depth']}
        return feed_dict

    def prediction_difference(self, data):
        """Evaluate prediction of the different individual branches for the given data.
        """
        keys = self.rgb_branch.keys()
        with self.graph.as_default():
            measures = [self.prediction]
            for tensors in (self.rgb_branch, self.depth_branch):
                for key in keys:
                    measures.append(tensors[key])
            outputs = self.sess.run(measures,
                                    feed_dict=self._evaluation_food(data))
        ret = {}
        ret['fused_label'] = outputs[0]
        i = 1
        for prefix in ('rgb', 'depth'):
            for key in keys:
                ret['{}_{}'.format(prefix, key)] = outputs[i]
                i = i + 1
        return ret
