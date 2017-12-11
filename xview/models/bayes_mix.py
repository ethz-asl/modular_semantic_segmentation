import tensorflow as tf
import numpy as np
from experiments.utils import ExperimentData

from .base_model import BaseModel
from xview.models.adapnet import adapnet
from xview.models.simple_fcn import encoder, decoder
from xview.models.simple_mix_fcn import bayes_fusion


class BayesMix(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers."""

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'learning_rate': 0.0,
        }
        standard_config.update(config)

        # load confusion matrices
        self.modalities = []
        self.confusion_matrices = {}
        for key, exp_id in config['eval_experiments'].values():
            self.modalities.append(key)
            self.confusion_matrices[key] = np.load(
                ExperimentData(exp_id)
                .get_artifact('confusion_matrix.npy')).astype(np.float32).T

        BaseModel.__init__(self, 'BayesMixture', output_dir=output_dir,
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
            if self.config['expert_model'] == 'adapnet':
                # Now we get the network output of the Adapnet expert.
                outputs = adapnet(inputs, prefix, self.config['num_units'],
                                  self.config['num_classes'], reuse=False)
            elif self.config['expert_model'] == 'fcn':
                outputs = encoder(inputs, prefix, self.config['num_units'],
                                  trainable=False, reuse=False)
                outputs.update(decoder(outputs['fused'], prefix,
                                       self.config['num_units'],
                                       self.config['num_classes'], 0.0, trainable=False,
                                       reuse=False))
            else:
                raise UserWarning('ERROR: Expert Model {} not found'
                                  .format(self.config['expert_model']))
            prob = tf.nn.softmax(outputs['score'])
            return prob

        rgb_prob = test_pipeline(self.test_X_rgb, 'rgb')
        depth_prob = test_pipeline(self.test_X_d, 'depth')

        rgb_label = tf.argmax(rgb_prob, 3, name='rgb_label_2d')
        depth_label = tf.argmax(depth_prob, 3, name='depth_label_2d')

        fused_score = bayes_fusion([rgb_label, depth_label],
                                   [self.confusion_matrices[x]
                                    for x in ['rgb', 'depth']],
                                   self.config)
        label = tf.argmax(fused_score, 3, name='label_2d')
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
