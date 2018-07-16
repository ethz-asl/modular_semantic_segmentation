import tensorflow as tf
import numpy as np
from experiments.utils import ExperimentData
from copy import deepcopy

from .dirichlet_fastfit import meanprecision_with_sufficient_statistic, \
                               fixedpoint_with_sufficient_statistic
#from .dirichletEstimation import findDirichletPriors
from .dirichletDifferentiation import findDirichletPriors
from .base_model import BaseModel, with_graph, transform_inputdata
from .basic_fusion_model import test_pipeline


def dirichlet_fusion(probs, conditionals, prior):
    """Bayes mixture by dirichlet conditionals.

    Args:
        probs: List of output class score tensors from the 2 experts
        conditionals: list of list of conditional dirichlet distributions, same order
            as probs
        prior: 1dim tensor with the prior probability of every class
    """
    num_classes = probs[0].shape[3]

    # We will collect all posteriors in this list
    log_likelihoods = []
    for i_expert in range(len(conditionals)):
        # compute p(expert output | groudn truth class x)
        log_likelihood = tf.stack(
            [conditionals[i_expert][c].log_prob(1e-20 + probs[i_expert])
             for c in range(num_classes)], axis=3)
        log_likelihoods.append(log_likelihood)

    fused_likelihood = tf.reduce_sum(tf.stack(log_likelihoods, axis=0), axis=0)

    return fused_likelihood + tf.log(1e-20 + prior)


class DirichletMix(BaseModel):
    """Mixture of CNN experts following the 'dirichlet mix' method.

    Args:
        num_units: number of intermediate feature neurons in the single expert
        num_classes: number of output classes
        expert_model: model o the CNN experts, either 'adapnet' or 'fcn'
        class_prior: either a string in ['data', 'uniform'] or a flaot between 0 and 1
            - 'data': prior is taken from the total class occurance probability in one
              of the confusion matrices
            - 'uniform': unform prior is assigned to all classes
            - float x: weighted sum x * uniform prior + (1-x) * data prior
        measurement_exp: If set, will load measurement of expert"s conditional output
            from this experiment.
            Otherwise you will have to call 'fit' before any inference is possible
    """

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'learning_rate': 0.0,
        }
        standard_config.update(config)

        self.modalities = config['modalities']

        # If specified, load mesaurements of the experts.
        if 'measurement_exp' in config or 'dirichlet_params' in config:
            if 'measurement_exp' in config:
                measurements = np.load(ExperimentData(config["measurement_exp"])
                                       .get_artifact("counts.npz"))
            else:
                measurements = config['dirichlet_params']
            self.dirichlet_params = {modality: measurements[modality].astype('float32')
                                     for modality in self.modalities}
            self.class_counts = measurements['class_counts'].astype('float32')
        else:
            print('WARNING: Could not yet import measurements, you need to fit this '
                  'model first.')

        BaseModel.__init__(self, 'DirichletFusion', output_dir=output_dir,
                           custom_training=True, **config)

    def _build_graph(self, train_data, test_data):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        num_classes = self.config['num_classes']

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        self.test_placeholders = {}
        for modality, channels in self.config['num_channels'].items():
            self.test_placeholders[modality] = tf.placeholder(
                tf.float32, shape=[None, None, None, channels])

        # The following part can only be build if measurements are already present.
        if hasattr(self, 'dirichlet_params'):

            probs = {modality: test_pipeline(test_data[modality], modality)['prob']
                     for modality in self.modalities}
            self.probs = {modality: probs[modality] /
                          tf.reduce_sum(probs[modality], axis=3, keep_dims=True)
                          for modality in self.modalities}

            # Create all the Dirichlet distributions conditional on ground-truth class
            dirichlets = {modality: {} for modality in self.modalities}

            sigma = self.config['sigma']

            for c in range(self.config['num_classes']):
                for m in self.modalities:
                    dirichlets[m][c] = tf.contrib.distributions.Dirichlet(
                        sigma * self.dirichlet_params[m][:, c].astype('float32'),
                        validate_args=False, allow_nan_stats=False)

            # Set the Prior of the classes
            uniform_prior = 1.0 / 14
            data_prior = (self.class_counts /
                          (1e-20 + self.class_counts.sum())).astype('float32')
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

            self.fused_score = dirichlet_fusion(self.probs.values(), dirichlets.values(),
                                                prior)

            label = tf.argmax(self.fused_score, 3, name='label_2d')
            self.prediction = label

            self.dirichlets = dirichlets

        else:
            # Build a training pipeline for measuring the differnet classifiers
            def train_pipeline(inputs, modality, labels):
                prob = test_pipeline(inputs, modality)['prob']

                stacked_labels = tf.stack([labels for _ in range(num_classes)], axis=3)

                eps = 1e-10
                sufficient_statistics = [
                    tf.log(tf.where(tf.equal(stacked_labels, c), eps + prob,
                                    tf.ones_like(prob)))
                    for c in range(num_classes)]
                combined = tf.stack([tf.reduce_sum(stat, axis=[0, 1, 2])
                                     for stat in sufficient_statistics], axis=0)
                return combined

            training_labels = train_data['labels']

            self.sufficient_statistics = {m: train_pipeline(train_data[m], m,
                                                            training_labels)
                                          for m in self.modalities}
            self.class_counts = tf.stack(
                [tf.reduce_sum(tf.cast(tf.equal(training_labels, c), tf.int64))
                 for c in range(num_classes)])

            # For compliance with base_model, we have to define a prediction outcome.
            # As we do not yet know how to do fusion, we simply set to 0
            self.prediction = 0
            self.fused_score = 0

        # To understand what"s going on under the hood, we expose a lot of intermediate
        # results for evaluation

    @transform_inputdata()
    @with_graph
    def _get_sufficient_statistic(self, data):
        """Generate a sufficient statistic of the given data to fit it later to a
        dirichlet model.

        Args:
            data: the data to fit, np.array or generator or tf.dataset
        Returns:
            statistics per modality aswell as class-counts
        """
        num_classes = self.config['num_classes']

        # store all measurements in these matrices
        counts = {m: np.zeros((num_classes, num_classes))
                  for m in self.modalities}
        class_counts = np.zeros(num_classes).astype('int64')

        # loop through the input data and create statistics
        while True:
            try:
                ops = [self.class_counts,
                       *[self.sufficient_statistics[m] for m in self.modalities]]
                new_counts = self.sess.run(ops, feed_dict={self.training_handle: data})
                class_counts += new_counts[0]
                i = 1
                for m in self.modalities:
                    counts[m] += new_counts[i]
                    i += 1
            except tf.errors.OutOfRangeError:
                break

        return counts, class_counts

    def _fit_sufficient_statistic(self, counts, class_counts):
        """Fit a dirichlet model to the given sufficient statistic."""
        num_classes = self.config['num_classes']

        # Now, given the sufficient statistic, run Expectation-Maximization to get the
        # Dirichlet parameters
        def dirichlet_em(measurements):
            """Find dirichlet parameters for all class-conditional dirichlets in
            measurements."""
            params = np.ones((num_classes, num_classes)).astype('float64')

            for c in range(num_classes):
                # Average the measurements over the encoutnered class examples to get the
                # sufficient statistic.
                if class_counts[c] == 0:
                    params[:, c] = np.ones(num_classes)
                    continue
                else:
                    ss = (measurements[c, :] / class_counts[c]).astype('float64')

                # sufficient statistic of negatives
                neg_ss = (measurements.sum(0) - measurements[c, :]) / \
                    (class_counts.sum() - class_counts[c])
                print(ss)
                print(class_counts[c])

                # The prior assumption is that all class output probabilities are equally
                # likely, i.e. all concentration parameters are 1
                prior = np.ones((num_classes)).astype('float64')

                #params[:, c] = meanprecision_with_sufficient_statistic(
                #    ss, class_counts[c], num_classes, prior, maxiter=10000,
                #    tol=1e-5, delta=self.config['delta'])
                #params[:, c] = fixedpoint_with_sufficient_statistic(
                #    ss, class_counts[c], num_classes, prior, maxiter=10000,
                #    tol=1e-5, delta=self.config['delta'])
                params[:, c] = findDirichletPriors(ss, neg_ss, prior, max_iter=10000,
                                                   delta=self.config['delta'],
                                                   beta=self.config['beta'])

                print('parameters for class {}: {}'.format(
                    c, ', '.join(['{}: {:.1f}'.format(i, params[i, c])
                                  for i in range(num_classes)])))
            return params

        self.dirichlet_params = {modality: dirichlet_em(counts[modality])
                                 for modality in self.modalities}
        self.class_counts = class_counts

        # Rebuild the graph with the new measurements:
        self._initialize_graph()

    def fit(self, data, *args, **kwargs):
        """Measure the encoder outputs against the given groundtruth for the given data.

        Args:
            data: standard data format: dict of np.arrays or tf.dataset
        """
        modality_counts, class_counts = self._get_sufficient_statistic(data)
        print('INFO: Measurements of classifiers finished, now EM')

        self._fit_sufficient_statistic(modality_counts, class_counts)
        print("INFO: MixFCN fitted to data")

        return_dict = deepcopy(self.dirichlet_params)
        return_dict['class_counts'] = self.class_counts
        return return_dict

    @transform_inputdata()
    def prediction_difference(self, data):
        """Evaluate prediction of the different individual branches for the given data.
        """
        keys = self.rgb_branch.keys()
        with self.graph.as_default():
            measures = [self.prediction, self.fused_score]
            for tensors in (self.rgb_branch, self.depth_branch):
                for key in keys:
                    measures.append(tensors[key])
            outputs = self.sess.run(measures, feed_dict={self.testing_handle: data})
        ret = {}
        ret['fused_label'] = outputs[0]
        ret['fused_score'] = outputs[1]
        i = 2
        for prefix in ('rgb', 'depth'):
            for key in keys:
                ret['{}_{}'.format(prefix, key)] = outputs[i]
                i = i + 1
        return ret
