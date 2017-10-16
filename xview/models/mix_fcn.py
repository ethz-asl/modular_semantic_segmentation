import tensorflow as tf
import numpy as np
from experiments.utils import ExperimentData

from .base_model import BaseModel
from xview.models.simple_fcn import encoder, decoder


def bayes_fusion(classifications, confusion_matrices):
    # We will collect all posteriors in this list
    log_posteriors = []

    for i_expert in range(len(confusion_matrices)):
        # compute p(expert output | groudn truth class x)
        confusion_matrix = confusion_matrices[i_expert]
        conditional = confusion_matrix / confusion_matrix.sum(0)
        # likelihood is conditional at the row of the output class
        likelihood = tf.gather(conditional, classifications[i_expert], axis=0)

        # set a uniform prior for all classes
        prior = 1.0 / 14

        log_posterior = tf.log(likelihood * prior /
                               tf.reduce_sum(likelihood * prior, axis=-1))
        log_posteriors.append(log_posterior)

    return tf.reduce_sum(tf.stack(log_posteriors, axis=0), axis=0)


class MixFCN(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers."""

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'learning_rate': 0.0,
            'batch_normalization': False
        }
        standard_config.update(config)

        # load confusion matrices
        rgb_confusion = np.load(
            ExperimentData(config['rgb_eval_experiment'])
            .get_artifact('confusion_matrix.npy')).astype(np.float32)
        depth_confusion = np.load(
            ExperimentData(config['depth_eval_experiment'])
            .get_artifact('confusion_matrix.npy')).astype(np.float32)

        self.confusion = {'rgb': rgb_confusion, 'depth': depth_confusion}

        BaseModel.__init__(self, 'FCN', output_dir=output_dir,
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

        batch_shape = tf.shape(self.test_X_rgb)

        def test_pipeline(inputs, prefix):
            # Now we get the network output of the FCN expert.
            features = encoder(inputs, prefix, self.config, reuse=False)
            score = decoder(features, prefix, 0.0, self.config, reuse=False)
            prob = tf.nn.softmax(score)
            return prob

        rgb_prob = test_pipeline(self.test_X_rgb, 'rgb')
        depth_prob = test_pipeline(self.test_X_d, 'depth')

        rgb_label = tf.argmax(rgb_prob, 3, name='rgb_label_2d')
        depth_label = tf.argmax(depth_prob, 3, name='depth_label_2d')

        fused_score = bayes_fusion([rgb_label, depth_label],
                                   [self.confusion[x] for x in ['rgb', 'depth']])
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
        # To understand what"s going on under the hood, we expose a lot of intermediate
        # results for evaluation
        self.rgb_branch = {'label': rgb_label}
        self.depth_branch = {'label': depth_label}

    def _enqueue_batch(self, batch, sess):
        # This model does not support training
        pass

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
