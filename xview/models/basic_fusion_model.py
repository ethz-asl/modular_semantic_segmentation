import tensorflow as tf
from copy import deepcopy

from xview.models.adapnet import adapnet
from xview.models.simple_fcn import fcn
from .base_model import BaseModel


def test_pipeline(inputs, prefix, **config):
    """Unified pipeline to produce semantic segmentation from the input with different
    network models. Currently FCN or Adapnet.
    """
    if config['expert_model'] == 'adapnet':
        # Now we get the network output of the Adapnet expert.
        outputs = adapnet(inputs, prefix, config['num_units'], config['num_classes'])
    elif config['expert_model'] == 'fcn':
        outputs = fcn(inputs, prefix, config['num_units'], config['num_classes'],
                      trainable=False, batchnorm=False)
    else:
        raise UserWarning('ERROR: Expert Model %s not found' % config['expert_model'])
    outputs['prob'] = tf.nn.softmax(outputs['score'])
    outputs['classification'] = tf.argmax(outputs['prob'], 3)
    return outputs


class FusionModel(BaseModel):
    """Mixture of CNN experts by averaging their score vectors.

    Args:
        name: Name of the Model
        num_units: number of intermediate feature neurons in the single expert
        num_classes: number of output classes
        expert_model: model o the CNN experts, either 'adapnet' or 'fcn'
        prefixes: dict, prefix of the variable names for the corresponding experts
    """

    def __init__(self, name=None, output_dir=None, **config):
        self.modalities = list(config['prefixes'].keys())

        BaseModel.__init__(self, name=name, output_dir=output_dir, custom_training=True,
                           **config)

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

    def _build_graph(self, train_data, test_data):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        self.expert_outputs = {m: test_pipeline(test_data[m], self.config['prefixes'][m],
                                                **self.config)
                               for m in self.modalities}
        self.prediction = self._fusion(self.expert_outputs)
