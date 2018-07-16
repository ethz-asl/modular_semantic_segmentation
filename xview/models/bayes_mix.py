import tensorflow as tf
import numpy as np
from itertools import product

from experiments.utils import ExperimentData

from .basic_fusion_model import FusionModel
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
    conditionals = []

    for i_expert in range(len(confusion_matrices)):
        # compute p(expert output | groudn truth class x)
        confusion_matrix = confusion_matrices[i_expert]
        conditional = np.nan_to_num(confusion_matrix / confusion_matrix.sum(0))
        conditionals.append(tf.gather(conditional, classifications[i_expert]))

        # likelihood is conditional at the row of the output class
        log_likelihoods.append(tf.log(1e-20 + tf.gather(conditional, classifications[i_expert])))

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

    return (tf.reduce_sum(tf.stack(log_likelihoods, axis=0), axis=0) + tf.log(prior),
            log_likelihoods,
            conditionals)


def bayes_decision_matrix(confusion_matrices, class_prior='data'):
    """
    Bayesian fusion for any combination of the expert's classification, as a simple lookup
    table. Can improve inference time.

    Args:
        confusion_matrices: list of numpoy arrays for every expert, order will be
            reflected in the dimensions of the lookup table
        class_prior: either a string in ['data', 'uniform'] or a flaot between 0 and 1
            - 'data': prior is taken from the total class occurance probability in one
              of the confusion matrices
            - 'uniform': unform prior is assigned to all classes
            - float x: weighted sum x * uniform prior + (1-x) * data prior
    Returns:
        a matrix of dimension <num_classes, ..., num classes>, where every matrix element
            is the corresponding fused classification.
    """
    num_classes = confusion_matrices[0].shape[0]
    num_experts = len(confusion_matrices)

    # all possible combinations of expert classifications
    expert_classifications = np.array(list(product(*(range(num_classes)
                                                     for _ in range(num_experts)))))
    log_likelihoods = np.zeros((expert_classifications.shape[0], num_experts,
                                num_classes))
    for i_expert in range(num_experts):
        # compute p(expert output | ground truth class x)
        confusion_matrix = confusion_matrices[i_expert]
        conditional = np.nan_to_num(confusion_matrix / confusion_matrix.sum(0))

        # likelihood is conditional at the row of the output class
        log_likelihoods[:, i_expert, :] = np.log(
            1e-20 + conditional[expert_classifications[:, i_expert]])

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

    # fused classification = argmax(sum of all log_likelihoods and log_prior)
    fused_classifications = np.argmax(log_likelihoods.sum(1) + np.log(prior), axis=1)
    # now reshape into requested format
    return fused_classifications.reshape([num_classes for _ in range(num_experts)])


class BayesFusion(FusionModel):
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
        self.confusion_matrices = {}
        if confusion_matrices:
            for key, matrix in confusion_matrices.items():
                self.modalities.append(key)
                self.confusion_matrices[key] = matrix.astype('float32').T
        else:
            for key, exp_id in config['eval_experiments'].items():

                self.confusion_matrices[key] = np.array(
                    ExperimentData(exp_id).get_record()['info']['confusion_matrix']
                    ['values']).astype('float32').T

        FusionModel.__init__(self, 'BayesFusion', output_dir=output_dir,
                             **standard_config)

    def _fusion(self, expert_outputs):
        fused_score, likelihoods, conditionals = bayes_fusion(
            [expert_outputs[m]['classification'] for m in self.modalities],
            [self.confusion_matrices[m] for m in self.modalities],
            self.config['class_prior'])
        # expose some diagnostics
        self.likelihoods = likelihoods
        self.conditionals = conditionals
        self.probs = {m: expert_outputs[m]['prob'] for m in self.modalities}
        return tf.argmax(fused_score, 3, name='label_2d')
