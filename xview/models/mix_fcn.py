import tensorflow as tf
import numpy as np
from experiments.utils import ExperimentData
from os import path
from types import GeneratorType
import threading

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
            num_classes = self.config['num_classes']

            # Build a training pipeline for measuring the differnet classifiers
            def train_pipeline(inputs, prefix, labels):
                # Now we get the network output of the FCN expert.
                outputs = {}
                outputs.update(encoder(inputs, prefix, self.config, reuse=True))
                outputs.update(decoder(outputs['fused'], prefix, 0.0, self.config,
                                       reuse=True))
                prob = tf.nn.softmax(outputs['score'])

                stacked_labels = tf.stack([labels for _ in range(num_classes)], axis=-1)

                eps = 1e-10
                sufficient_statistics = [tf.log(eps + tf.where(stacked_labels == c, prob,
                                                              tf.ones_like(prob)))
                                        for c in range(num_classes)]
                combined = tf.stack([tf.reduce_sum(stat, axis=[0, 1, 2])
                                     for stat in sufficient_statistics], axis=0)
                return combined

            self.train_rgb = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.train_depth = tf.placeholder(tf.float32, shape=[None, None, None, 1])
            self.train_Y = tf.placeholder(tf.float32, shape=[None, None, None])
            # An input queue is defined to load the data for several batches in advance and
            # keep the gpu as busy as we can.
            # IMPORTANT: The size of this queue can grow big very easily with growing
            # batchsize, therefore do not make the queue too long, otherwise we risk getting
            # killed by the OS
            q = tf.FIFOQueue(3, [tf.float32, tf.float32, tf.float32])
            self.enqueue_op = q.enqueue([self.train_rgb, self.train_depth, self.train_Y])
            train_rgb, train_depth, training_labels = q.dequeue()
            # The queue output does not have a defined shape, so we have to define it here to
            # be compatible with tf.layers.
            train_rgb.set_shape([None, None, None, 3])
            train_depth.set_shape([None, None, None, 1])

            # This operation has to be called to close the input queue and free the space it
            # occupies in memory.
            self.close_queue_op = q.close(cancel_pending_enqueues=True)
            self.queue_is_empty_op = tf.equal(q.size(), 0)
            # To support tensorflow 1.2, we have to set this flag manually.
            self.queue_is_closed = False

            self.rgb_sufficient_statistic = train_pipeline(train_rgb, 'rgb',
                                                           training_labels)
            self.depth_sufficient_statistic = train_pipeline(train_depth, 'depth',
                                                             training_labels)
            self.class_counts = tf.stack(
                [tf.reduce_sum(tf.cast(tf.equal(training_labels, c), tf.int16))
                 for c in range(num_classes)])

            # For compliance with base_model, we have to define a prediction outcome.
            # As we do not yet know how to do fusion, we simply take rgb.
            self.prediction = rgb_label

        # To understand what"s going on under the hood, we expose a lot of intermediate
        # results for evaluation
        self.rgb_branch = {'label': rgb_label}
        self.depth_branch = {'label': depth_label}

    def fit(self, data):
        """Measure the encoder outputs against the given groundtruth for the given data.

        Args:
            data: As usual, an instance of DataWrapper
        """
        num_classes = self.config['num_classes']

        with self.graph.as_default():
            # store all measurements in these matrices
            rgb_measurements = np.zeros((num_classes, num_classes))
            depth_measurements = np.zeros((num_classes, num_classes))
            class_counts = np.zeros(num_classes)

            # Create a thread to load data.
            coord = tf.train.Coordinator()
            t = threading.Thread(target=self._load_and_enqueue,
                                 args=(self.sess, data))
            t.start()

            queue_empty = False
            while not (queue_empty and self.queue_is_closed):
                new_rgb, new_depth, new_count = self.sess.run(
                    [self.rgb_sufficient_statistic, self.depth_sufficient_statistic,
                     self.class_counts])
                rgb_measurements += new_rgb
                depth_measurements += new_depth
                class_counts += new_count
                queue_empty = self.sess.run(self.queue_is_empty_op)

            coord.join([t])

        print('INFO: Measurements of classifiers finished, now EM')

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

    def _enqueue_batch(self, batch, sess):
        with self.graph.as_default():
            feed_dict = {self.train_rgb: batch['rgb'], self.train_depth: batch['depth'],
                         self.train_Y: batch['labels']}
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def _load_and_enqueue(self, sess, data):
        """Internal handler method for the input data queue. Will run in a seperate
        thread.

        Overwritten from base_model because we only want to go once through the data.

        Args:
            sess: The current session, needs to be the same as in the main thread.
            data: The data to load. See method predict for specifications.
        """
        with self.graph.as_default():
            # We enqueue new data until it tells us to stop.
            try:
                if isinstance(data, GeneratorType):
                    for batch in data:
                        self._enqueue_batch(batch, sess)
                else:
                    self._enqueue_batch(data, sess)
            except tf.errors.CancelledError:
                print('INFO: Input queue is closed, cannot enqueue any more data.')
            sess.run(self.close_queue_op)
            self.queue_is_closed = True

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
