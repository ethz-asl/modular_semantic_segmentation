import tensorflow as tf

from .basic_fusion_model import FusionModel


class AverageFusion(FusionModel):
    """Mixture of CNN experts by averaging their score vectors.

    Args:
        num_units: number of intermediate feature neurons in the single expert
        num_classes: number of output classes
        expert_model: model o the CNN experts, either 'adapnet' or 'fcn'
    """

    def __init__(self, output_dir=None, **config):
        FusionModel.__init__(self, 'AverageFusion', output_dir=output_dir, **config)

    def _fusion(self, expert_outputs):
        average_prob = tf.reduce_mean(tf.stack([expert_outputs[m]['prob']
                                                for m in self.modalities]), axis=0)
        return tf.argmax(average_prob, axis=3)
