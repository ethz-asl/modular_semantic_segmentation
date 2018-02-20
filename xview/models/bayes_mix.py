import tensorflow as tf
import numpy as np
from experiments.utils import ExperimentData

from .base_model import BaseModel
from xview.models.adapnet import adapnet
from xview.models.simple_fcn import encoder, decoder


def bayes_fusion(classifications, confusion_matrices, class_prior='data'):
    """Bayesian fusion of the classifications based on the classifiers confusion
    matrices.

    Args:
        classifications: list of batch tensors for every expert in the dimension
            <batchsize, image dim1, image dim2>
        confusion_matrices: list of numpy arrays for every expert, same order as
            classifications. Confusion matrices need dimension <num classes, num classes>
        class_prior: either a string in ['data', 'uniform'] or a flaot between 0 and 1
            - 'data': prior is taken from the total class occurance probability in one
              of the confusion matrices
            - 'uniform': unform prior is assigned to all classes
            - float x: weighted sum x * uniform prior + (1-x) * data prior
    Returns:
        class score tensor of dimension <batchsize, image dim1, image dim2, num classes>
    """
    # We will collect all posteriors in this list
    log_likelihoods = []

    for i_expert in range(len(confusion_matrices)):
        # compute p(expert output | groudn truth class x)
        confusion_matrix = confusion_matrices[i_expert]
        conditional = np.nan_to_num(confusion_matrix / confusion_matrix.sum(0))

        # likelihood is conditional at the row of the output class
        log_likelihoods.append(tf.log(tf.gather(conditional, classifications[i_expert])))

    uniform_prior = 1.0 / 14
    data_prior = confusion_matrix.sum(0) / confusion_matrix.sum()
    if class_prior == 'uniform':
        # set a uniform prior for all classes
        prior = uniform_prior
    elif class_prior == 'data':
        prior = data_prior
    else:
        # The class_prior parameter is now considered a weight for the mixture
        # between both priors.
        weight = float(class_prior)
        prior = weight * uniform_prior + (1 - weight) * data_prior
        prior = prior / prior.sum()

    return tf.reduce_sum(tf.stack(log_likelihoods, axis=0), axis=0) + tf.log(prior)


class BayesMix(BaseModel):
    """Mixture of CNN experts following the 'bayes mix' method.

    Args:
        num_units: number of intermediate feature neurons in the single expert
        num_classes: number of output classes
        confusion_matrices: if set, should contain a list of confusion matrices for both
            experts
        eval_experiments: If confusion_matrices is not set, the matrices are loaded from
            this list of experiment ids
        expert_model: model o the CNN experts, either 'adapnet' or 'fcn'
    """

    def __init__(self, output_dir=None, confusion_matrices=False, **config):
        standard_config = {
            'learning_rate': 0.0,
            'class_prior': 'data'
        }
        standard_config.update(config)

        # load confusion matrices
        self.modalities = []
        if confusion_matrices:
            self.confusion_matrices = confusion_matrices
        else:
            self.confusion_matrices = {}
            for key, exp_id in config['eval_experiments'].items():
                self.modalities.append(key)
                self.confusion_matrices[key] = np.array(
                    ExperimentData(exp_id).get_record()['info']['confusion_matrix']
                    ['values']).astype('float32').T

        BaseModel.__init__(self, 'BayesMixture', output_dir=output_dir,
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

        probs = {modality: test_pipeline(self.test_placeholders[modality],
                                         self.config['prefixes'][modality])
                 for modality in self.modalities}

        fused_score = bayes_fusion([tf.argmax(prob, 3) for prob in probs.values()],
                                   [self.confusion_matrices[x]
                                    for x in self.modalities],
                                   self.config['class_prior'])
        label = tf.argmax(fused_score, 3, name='label_2d')
        self.prediction = label

    def _enqueue_batch(self, batch, sess):
        # This model does not support training
        pass

    def _evaluation_food(self, data):
        feed_dict = {self.test_placeholders[modality]: data[modality]
                     for modality in self.modalities}
        return feed_dict
