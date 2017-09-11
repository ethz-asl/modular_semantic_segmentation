import tensorflow as tf
import numpy as np
from experiments.utils import ExperimentData

from .base_model import BaseModel
from xview.models import SimpleFCN


class MixFCN(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers."""

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'learning_rate': 0.0,
            'batch_normalization': False
        }
        standard_config.update(config)
        self.fcn = SimpleFCN(num_channels=0, modality='none', **standard_config)

        # load confusion matrices
        rgb_confusion = np.load(
            ExperimentData(100).get_artifact('confusion_matrix.npy')).astype(np.float32)
        depth_confusion = np.load(
            ExperimentData(98).get_artifact('confusion_matrix.npy')).astype(np.float32)

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
            # Load the confusion matrix of this expert.
            confusion = self.confusion[prefix]
            # Norm it to rows summing up to 1 and bring it into the same shape as the
            # batches we evaluate.
            normed_confusion = confusion / confusion.sum(1)
            normed_confusion = tf.tile(tf.reshape(normed_confusion, [1, 1, 1, 14, 14]),
                                       [batch_shape[0], batch_shape[1], batch_shape[2],
                                        1, 1])

            # Now we get the network output of the FCN expert.
            features = self.fcn._encoder(inputs, prefix, reuse=False)
            score = self.fcn._decoder(features, prefix, 0.0, reuse=False)
            prob = tf.nn.softmax(score)
            # Based on the confusion matrix, we find the believe about the pixel class by
            # multiplying the expert output e(x) with the probability of p(x in C|e(x)),
            # which we get from the normed confusion matrix above.
            believe = tf.tile(tf.expand_dims(prob, axis=-2), [1, 1, 1, 14, 1])
            believe = tf.reduce_sum(normed_confusion * believe, axis=-1)
            return prob, believe

        rgb_prob, rgb_bel = test_pipeline(self.test_X_rgb, 'rgb')
        depth_prob, depth_bel = test_pipeline(self.test_X_d, 'depth')

        # Based on the relative precision of the experts and their output probabilities,
        # we find a weight per expert and pixel.
        rgb_precision = np.diag(self.confusion['rgb']) / self.confusion['rgb'].sum(1)
        depth_precision = np.diag(self.confusion['depth']) / \
            self.confusion['depth'].sum(1)
        relative_precision = np.stack([rgb_precision, depth_precision], axis=0)
        relative_precision = relative_precision / relative_precision.sum(0)
        # Now reshape this precision according to batch shape
        relative_precision = np.reshape(relative_precision, (2, 1, 1, 14))
        relative_rgb_precision = tf.tile(relative_precision[:1], [batch_shape[0],
                                         batch_shape[1], batch_shape[2], 1])
        relative_depth_precision = tf.tile(relative_precision[1:], [batch_shape[0],
                                           batch_shape[1], batch_shape[2], 1])
        rgb_weight = tf.reduce_sum(rgb_prob * relative_rgb_precision, axis=3,
                                   keep_dims=True)
        depth_weight = tf.reduce_sum(depth_prob * relative_depth_precision, axis=3,
                                     keep_dims=True)

        # Now we take the weighted sum over all expert believes as network output.
        # The 'prior' of each expert is the previously calculated weight stretched out by
        # the separation factor, which will increase the weight-difference between the
        # experts for higher values.
        separation_factor = float(self.config['separation_factor'])
        fused_score = rgb_bel * tf.exp(rgb_weight * separation_factor) + \
            depth_bel * tf.exp(depth_weight * separation_factor)

        label = tf.argmax(fused_score, 3, name='label_2d')

        # We also want to make predictions in the absense of one sensor modality.
        # Therefore, we produce the standard classifications of each single expert.
        rgb_label = tf.argmax(rgb_prob, 3, name='rgb_label_2d')
        depth_label = tf.argmax(depth_prob, 3, name='depth_label_2d')

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
        self.rgb_branch = {'label': rgb_label,
                           'believe': rgb_bel,
                           'relative_precision': relative_rgb_precision}
        self.depth_branch = {'label': depth_label,
                             'believe': depth_bel,
                             'relative_precision': relative_depth_precision}

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
