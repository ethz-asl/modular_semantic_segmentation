import tensorflow as tf

from .base_model import BaseModel
from xview.models.adapnet import adapnet
from xview.models.simple_fcn import encoder, decoder


class AverageMix(BaseModel):
    """Mixture of CNN experts by averaging their score vectors.

    Args:
        num_units: number of intermediate feature neurons in the single expert
        num_classes: number of output classes
        expert_model: model o the CNN experts, either 'adapnet' or 'fcn'
    """

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'learning_rate': 0.0,
        }
        standard_config.update(config)

        # load confusion matrices
        self.modalities = []

        BaseModel.__init__(self, 'AverageMixture', output_dir=output_dir,
                           supports_training=False, **config)

    def _build_graph(self):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        self.test_placeholders = {}
        for modality, channels in self.config['num_channels'].items():
            self.test_placeholders[modality] = tf.placeholder(
                tf.float32, shape=[None, None, None, channels])

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

        fused_score = tf.reduce_mean([test_pipeline(self.test_placeholders[m],
                                                    self.config['prefixes'][m])
                                      for m in self.modalities], axis=0)

        label = tf.argmax(fused_score, 3, name='label_2d')
        self.prediction = label

    def _enqueue_batch(self, batch, sess):
        # This model does not support training
        pass

    def _evaluation_food(self, data):
        feed_dict = {self.test_placeholders[modality]: data[modality]
                     for modality in self.modalities}
        return feed_dict
