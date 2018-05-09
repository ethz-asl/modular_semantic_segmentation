"""Evaluation of trained uncertainty models."""
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from xview.models import get_model
from xview.datasets import get_dataset
from sys import stdout

from .utils import get_mongo_observer
from .evaluation import import_weights_into_network


def evaluate_misclassification_detection(net, data, metric, print_results=True):
    tpr, fpr, auroc, thresholds = net.misclassification_detection_score(
        data.get_testset(), metric)
    if print_results:
        print('Misclassification Detection of {} on {} with {} metric'.format(
            net.name, type(data).__name__, metric))
        print('AUROC {:.3f}'.format(auroc))
        stdout.flush()
    return {'TPR': tpr, 'FPR': fpr, 'AUROC': auroc, 'thresholds': thresholds}


ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_mongo_observer())


@ex.automain
def misclassification_detection(modelname, net_config, dataset, starting_weights,
                                uncertainty_metrics, _run):
    model = get_model(modelname)
    data = get_dataset(dataset['name'])
    with model(data_description=data.get_data_description(), **net_config) as net:
        data = data(**dataset)
        import_weights_into_network(net, starting_weights)
        for metric in uncertainty_metrics:
            measurements = evaluate_misclassification_detection(net, data, metric)
            _run.info.setdefault('measurements', {})[metric] = measurements
