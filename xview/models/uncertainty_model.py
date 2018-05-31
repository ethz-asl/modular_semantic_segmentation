import tensorflow as tf
import numpy as np
from sklearn.metrics import auc
from .base_model import BaseModel, transform_inputdata, with_graph
from .dirichletEstimation import findDirichletPriors


def make_roc(values_for_positives, values_for_negatives, num_thresholds=100):
    min_val = np.min(values_for_positives.min(), values_for_negatives.min())
    max_val = np.max(values_for_positives.max(), values_for_negatives.max())
    thresholds = np.linspace(min_val, max_val, num=num_thresholds)
    tpr = np.zeros(num_thresholds)
    fpr = np.zeros(num_thresholds)
    n_positives = values_for_positives.shape[0]
    n_negatives = values_for_negatives.shape[0]
    for i in range(num_thresholds):
        tp = np.sum(values_for_positives > thresholds[i])
        fp = np.sum(values_for_negatives > thresholds[i])
        tpr[i] = tp / n_positives
        fpr[i] = fp / n_negatives

    # area under the curve
    auroc = auc(fpr, tpr, reorder=True)
    return fpr, tpr, auroc, thresholds


class UncertaintyModel(BaseModel):

    @transform_inputdata()
    @with_graph
    def misclassification_detection_score(self, data, uncertainty_attr,
                                          num_thresholds=100):
        uncertainty_attr = getattr(self, uncertainty_attr)

        uncertainty_for_correct_classes = []
        uncertainty_for_misclassification = []
        while True:
            try:
                # get classification and uncertainty for given dataset
                gt, prediction, uncertainty = self.sess.run(
                    [self.evaluation_labels, self.prediction, uncertainty_attr],
                    feed_dict={self.testing_handle: data})
                gt = gt.flatten()
                prediction = prediction.flatten()
                uncertainty = uncertainty.flatten()
                uncertainty_for_correct_classes.append(uncertainty[gt == prediction])
                uncertainty_for_misclassification.append(
                    uncertainty[gt != prediction])
            except tf.errors.OutOfRangeError:
                break
        # reshape uncertainties into 1 big vector
        uncertainty_for_misclassification = np.concatenate(
            uncertainty_for_misclassification).flatten()
        uncertainty_for_correct_classes = np.concatenate(
            uncertainty_for_correct_classes).flatten()

        # now calculate ROC
        return make_roc(uncertainty_for_misclassification,
                        uncertainty_for_correct_classes,
                        num_thresholds=num_thresholds)

    @transform_inputdata()
    @with_graph
    def out_of_distribution_detection_score(self, data, uncertainty_attr,
                                            num_thresholds=500):
        uncertainty_attr = getattr(self, uncertainty_attr)

        uncertainty_in_distribution = []
        uncertainty_out_of_distribution = []

        while True:
            try:
                # get classification and uncertainty for given dataset
                gt, uncertainty = self.sess.run(
                    [self.evaluation_labels, uncertainty_attr],
                    feed_dict={self.testing_handle: data})
                gt = gt.flatten()
                uncertainty = uncertainty.flatten()
                uncertainty_in_distribution.append(uncertainty[gt == 0])
                uncertainty_out_of_distribution.append(uncertainty[gt == 1])
            except tf.errors.OutOfRangeError:
                break
        # reshape uncertainties into 1 big vector
        uncertainty_in_distribution = np.concatenate(
            uncertainty_in_distribution).flatten()
        uncertainty_out_of_distribution = np.concatenate(
            uncertainty_out_of_distribution).flatten()

        # now calculate ROC
        return make_roc(uncertainty_out_of_distribution, uncertainty_in_distribution,
                        num_thresholds=num_thresholds)

    @transform_inputdata()
    @with_graph
    def ambiguity_detection_score(self, data, uncertainty_attr, ambiguous_classes,
                                  num_thresholds=500):
        uncertainty_attr = getattr(self, uncertainty_attr)

        ambiguous_uncertainties = []
        nonambiguous_uncertainties = []
        while True:
            try:
                # get classification and uncertainty for given dataset
                pred, uncertainty = self.sess.run([self.prediction, uncertainty_attr],
                                                  feed_dict={self.testing_handle: data})
                for c in range(pred.max()):
                    if c in ambiguous_classes:
                        ambiguous_uncertainties.append(uncertainty[pred == c])
                    else:
                        nonambiguous_uncertainties.append(uncertainty[pred == c])
            except tf.errors.OutOfRangeError:
                break
        # reshape uncertainties into 1 big vector
        ambiguous_uncertainties = np.concatenate(ambiguous_uncertainties).flatten()
        nonambiguous_uncertainties = np.concatenate(nonambiguous_uncertainties).flatten()

        # now calculate ROC
        return make_roc(ambiguous_uncertainties, nonambiguous_uncertainties,
                        num_thresholds=num_thresholds)

    @transform_inputdata()
    @with_graph
    def nll_score(self, data):
        per_class_nll = np.zeros(self.config['num_classes'])
        class_count = np.zeros(self.config['num_classes'])
        while True:
            try:
                gt, prob, pred = self.sess.run(
                    [self.evaluation_labels, self.prob, self.prediction],
                    feed_dict={self.testing_handle: data})
                log_prob = np.log(prob)
                for c in range(self.config['num_classes']):
                    nll = log_prob[np.logical_and(gt == pred, gt == c)].sum()
                    nll += (1 - log_prob[np.logical_and(gt == c, gt != pred)]).sum()
                    per_class_nll[c] += nll
                    class_count[c] += np.sum(gt == c)
            except tf.errors.OutOfRangeError:
                break
        return per_class_nll / class_count, class_count

    @transform_inputdata()
    @with_graph
    def value_distribution(self, data, uncertainty_attr, num_bins=20):
        uncertainty_attr = getattr(self, uncertainty_attr)

        values = [[[] for _ in range(self.config['num_classes'])]
                  for _ in range(self.config['num_classes'])]
        while True:
            try:
                gt, pred, uncertainty = self.sess.run(
                    [self.evaluation_labels, self.prediction, uncertainty_attr],
                    feed_dict={self.testing_handle: data})
                for t in range(self.config['num_classes']):
                    for c in range(self.config['num_classes']):
                        values[t][c].append(uncertainty[np.logical_and(gt == t,
                                                                       pred == c)])
            except tf.errors.OutOfRangeError:
                break
        values = [[np.concatenate(cell) for cell in row] for row in values]
        return [[{'bins': np.histogram(cell, bins=num_bins), 'mean': cell.mean(),
                  'var': cell.var()} for cell in row] for row in values]

    @transform_inputdata()
    @with_graph
    def prob_distribution(self, data):
        nc = self.config['num_classes']
        sufficient_statistics = [[np.zeros(nc) for _ in range(nc)] for _ in range(nc)]
        class_counts = [[0 for _ in range(nc)] for _ in range(nc)]
        prob_sum = [[np.zeros(nc) for _ in range(nc)] for _ in range(nc)]
        while True:
            try:
                gt, pred, prob = self.sess.run(
                    [self.evaluation_labels, self.prediction, self.prob],
                    feed_dict={self.testing_handle: data})
                for t in range(nc):
                    for c in range(nc):
                        ss = np.log(prob[np.logical_and(gt == t, pred == c)]).sum(0)
                        sufficient_statistics[t][c] += ss
                        class_counts[t][c] += np.logical_and(gt == t, pred == c).sum()
                        prob_sum[t][c] += prob[np.logical_and(gt == t, pred == c)].sum(0)
            except tf.errors.OutOfRangeError:
                break
        # take the mean over the sufficient statistics
        sufficient_statistics = [[sufficient_statistics[t][c] / class_counts[t][c]
                                  for c in range(nc)] for t in range(nc)]
        try:
            dirichlets = [[findDirichletPriors(sufficient_statistics[t][c], np.ones(nc))
                           for c in range(nc)] for t in range(nc)]
        except OverflowError:
            dirichlets = [[np.array([np.nan for _ in range(nc)])
                           for c in range(nc)] for t in range(nc)]
        mean_prob = [[prob_sum[t][c] / class_counts[t][c]
                      for c in range(nc)] for t in range(nc)]
        return dirichlets, mean_prob

    @transform_inputdata()
    @with_graph
    def mean_diff(self, data, prior, condition=lambda t, c: True):
        diff = 0
        n = 0
        while True:
            try:
                gt, pred, prob = self.sess.run(
                    [self.evaluation_labels, self.prediction, self.prob],
                    feed_dict={self.testing_handle: data})

                cond = condition(gt, pred)
                diff += np.sum(np.where(cond,
                                        np.sum(np.abs(prob - prior), axis=-1),
                                        np.zeros_like(pred)))
                n += np.sum(np.where(cond,
                                     np.ones_like(pred),
                                     np.zeros_like(pred)))
            except tf.errors.OutOfRangeError:
                break

        return diff / n
