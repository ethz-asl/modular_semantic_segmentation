import tensorflow as tf

from .base_model import BaseModel
from xview.models.simple_fcn import fcn


def variance_fusion(probs, variances):
    # 'certainty' is used as a pseudonym for 1/sigma^2
    # we have to expand the last dimension so all class scores are multiplied with the
    # same variance per pixel
    certainties = tf.stack([1 / (1e-20 + variance) for variance in variances], axis=0)
    probs = tf.stack(probs, axis=0)

    return tf.reduce_sum(certainties * probs, axis=0) / \
        tf.reduce_sum(certainties, axis=0)


class VarianceFusion(BaseModel):

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'learning_rate': 0.0,
        }
        standard_config.update(config)

        self.modalities = config['modalities']

        BaseModel.__init__(self, 'VarianceMixture', output_dir=output_dir,
                           supports_training=False, **config)

    def _build_graph(self, train_data, test_data):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Network for testing / evaluation
        def get_prob(inputs, modality):
            prefix = self.config['prefixes'][modality]

            layers = fcn(inputs, prefix, self.config['num_units'],
                         self.config['num_classes'], trainable=False,
                         is_training=False, dropout_rate=0, dropout_layers=[],
                         batchnorm=False)
            prob = tf.nn.softmax(layers['score'])
            return prob

        def test_pipeline(inputs, modality):

            def sample_pipeline(inputs, modality, reuse=False):
                prefix = self.config['prefixes'][modality]

                assert self.config['expert_model'] == 'fcn'

                layers = fcn(inputs, prefix, self.config['num_units'],
                             self.config['num_classes'], trainable=False,
                             is_training=False, dropout_rate=self.config['dropout_rate'],
                             dropout_layers=['pool3'], batchnorm=False)
                prob = tf.nn.softmax(layers['score'])
                return prob

            # For classification, we sample distributions with Dropout-Monte-Carlo and
            # fuse output according to variance
            samples = tf.stack([sample_pipeline(inputs, modality, reuse=(i != 0))
                                for i in range(self.config['num_samples'])], axis=4)

            variance = tf.reduce_mean(tf.nn.moments(samples, [4])[1], axis=3,
                                      keep_dims=True)

            # We get the label by passing the input without dropout
            return get_prob(inputs, modality), variance

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
