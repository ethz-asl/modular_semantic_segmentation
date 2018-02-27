import tensorflow as tf
from tensorflow.python.layers.layers import dropout

from .base_model import BaseModel
from xview.models.adapnet import adapnet
from xview.models.simple_fcn import encoder, decoder


def variance_fusion(probs, variances):
    # 'certainty' is used as a pseudonym for 1/sigma^2
    # we have to expand the last dimension so all class scores are multiplied with the
    # same variance per pixel
    certainties = tf.stack([1 / (1e-20 + variance) for variance in variances], axis=0)
    probs = tf.stack(probs, axis=0)

    print('variance', variances[0].shape)
    print('certainties', certainties.shape)
    print('probs', probs.shape)

    return tf.reduce_sum(certainties * probs, axis=0) / \
        tf.reduce_sum(certainties, axis=0)


class VarianceMix(BaseModel):

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'learning_rate': 0.0,
        }
        standard_config.update(config)

        self.modalities = config['modalities']

        BaseModel.__init__(self, 'VarianceMixture', output_dir=output_dir,
                           supports_training=False, **config)

    def _build_graph(self):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        # rgb channel
        self.test_placeholders = {}
        for modality, channels in self.config['num_channels'].items():
            self.test_placeholders[modality] = tf.placeholder(
                tf.float32, shape=[None, None, None, channels])

        def get_prob(inputs, modality, reuse=False):
            prefix = self.config['prefixes'][modality]

            if self.config['expert_model'] == 'adapnet':
                # Now we get the network output of the Adapnet expert.
                outputs = adapnet(inputs, prefix, self.config['num_units'],
                                  self.config['num_classes'], reuse=reuse)
            elif self.config['expert_model'] == 'fcn':
                outputs = encoder(inputs, prefix, self.config['num_units'],
                                  trainable=False, reuse=reuse)
                outputs.update(decoder(outputs['fused'], prefix,
                                       self.config['num_units'],
                                       self.config['num_classes'],
                                       trainable=False, reuse=reuse))
            else:
                raise UserWarning('ERROR: Expert Model {} not found'
                                  .format(self.config['expert_model']))
            prob = tf.nn.softmax(outputs['score'])
            return prob

        def test_pipeline(inputs, modality):

            def sample_pipeline(inputs, modality, reuse=False):
                prefix = self.config['prefixes'][modality]

                # We apply dropout at the input.
                # We do want to set whole pixels to 0, therefore out noise-shape has
                # dim 1 for the channel-space:
                #input_shape = tf.shape(inputs)
                #noise_shape = [input_shape[0], input_shape[1], input_shape[2], 1]
                #inputs = dropout(inputs, rate=self.config['dropout_rate'], training=True,
                #                 noise_shape=noise_shape,
                #                 name='{}_dropout'.format(prefix))
                assert self.config['expert_model'] == 'fcn'

                outputs = encoder(inputs, prefix, self.config['num_units'],
                                  self.config['dropout_rate'], trainable=False,
                                  is_training=True, reuse=reuse)
                outputs.update(decoder(outputs['fused'], prefix,
                                       self.config['num_units'],
                                       self.config['num_classes'],
                                       trainable=False, reuse=reuse))
                prob = tf.nn.softmax(outputs['score'])
                return prob

            # For classification, we sample distributions with Dropout-Monte-Carlo and
            # fuse output according to variance
            samples = tf.stack([sample_pipeline(inputs, modality, reuse=(i != 0))
                                for i in range(self.config['num_samples'])], axis=4)

            variance = tf.reduce_mean(tf.nn.moments(samples, [4])[1], axis=3,
                                      keep_dims=True)

            # We get the label by passing the input without dropout
            return get_prob(inputs, modality, reuse=True), variance

        probs = {}
        vars = {}
        for modality in self.modalities:
            probs[modality], vars[modality] = test_pipeline(
                self.test_placeholders[modality], modality)

        self.probs = {modality: probs[modality] /
                      tf.reduce_sum(probs[modality], axis=3, keep_dims=True)
                      for modality in self.modalities}

        self.fused_score = variance_fusion(probs.values(), vars.values())
        label = tf.argmax(self.fused_score, 3, name='label_2d')
        self.prediction = label

    def _evaluation_food(self, data):
        feed_dict = {self.test_placeholders[modality]: data[modality]
                     for modality in self.modalities}
        return feed_dict

    def _enqueue_batch(self, batch, sess):
        # This model does not support training
        pass
