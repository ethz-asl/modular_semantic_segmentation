import tensorflow as tf

from xview.models.adapnet import adapnet
from xview.models.simple_fcn import encoder, decoder
from .base_model import BaseModel


class FusionModel(BaseModel):
    """Mixture of CNN experts by averaging their score vectors.

    Args:
        name: Name of the Model
        num_units: number of intermediate feature neurons in the single expert
        num_classes: number of output classes
        expert_model: model o the CNN experts, either 'adapnet' or 'fcn'
        prefixes: dict, prefix of the variable names for the corresponding experts
    """

    def __init__(self, name, output_dir=None, **config):
        self.modalities = list(config['prefixes'].keys())

        BaseModel.__init__(self, name, output_dir=output_dir, supports_training=False,
                           learning_rate=0.0, **config)

    def _fusion(self, expert_outputs):
        """Implements the fusion based on expert outputs.

        Args:
            expert_outputs: dict {modality: layers} where layers is a dict of all layer
                intermediate outputs, including 'score', the softmax output 'prob' and
                the expert's classification 'classification'.
        Returns:
            List of tensors that will then be assigned to self.fusion_result.
            The first tensor has to be the fused classification.
        """
        raise NotImplementedError

    def _test_pipeline(self, inputs, prefix):
        if self.config['expert_model'] == 'adapnet':
            # Now we get the network output of the Adapnet expert.
            outputs = adapnet(inputs, prefix, self.config['num_units'],
                              self.config['num_classes'], reuse=False)
        elif self.config['expert_model'] == 'fcn':
            outputs = encoder(inputs, prefix, self.config['num_units'], 0.0,
                              trainable=False, reuse=False)
            outputs.update(decoder(outputs['fused'], prefix,
                                   self.config['num_units'],
                                   self.config['num_classes'], trainable=False,
                                   reuse=False))
        else:
            raise UserWarning('ERROR: Expert Model {} not found'
                              .format(self.config['expert_model']))
        outputs['prob'] = tf.nn.softmax(outputs['score'])
        outputs['classification'] = tf.argmax(outputs['prob'], 3)
        return outputs

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

        self.expert_outputs = {m: self._test_pipeline(self.test_placeholders[m],
                                                      self.config['prefixes'][m])
                               for m in self.modalities}
        self.fusion_result = self._fusion(self.expert_outputs)

        self.prediction = self.fusion_result[0]

    def _enqueue_batch(self, batch, sess):
        # This model does not support training
        pass

    def _evaluation_food(self, data):
        feed_dict = {self.test_placeholders[modality]: data[modality]
                     for modality in self.modalities}
        return feed_dict

    def get_insight(self, data):
        """Run the network on the given data and get all the intermediate fusion outputs,
        aswell as the experts probs and scores."""
        with self.graph.as_default():
            probs, scores, fusion_results = \
                self.sess.run([[self.expert_outputs[m]['prob'] for m in self.modalities],
                               [self.expert_outputs[m]['score'] for m in self.modalities],
                               self.fusion_result],
                              feed_dict=self._evaluation_food(data))
        return fusion_results, probs, scores
