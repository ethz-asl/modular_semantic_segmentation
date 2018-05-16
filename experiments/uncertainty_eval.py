"""Evaluation of trained uncertainty models."""
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from xview.models import get_model
from xview.datasets import get_dataset
from sys import stdout

from .utils import get_mongo_observer
from .evaluation import import_weights_into_network
from .different_evaluation_parameters import grid_search


def evaluate_uncertainty(net, data, metric, benchmark='misclassification',
                         print_results=True):
    if benchmark == 'misclassification':
        fpr, tpr, auroc, thresholds = net.misclassification_detection_score(data, metric)
    elif benchmark == 'out_of_distribution':
        fpr, tpr, auroc, thresholds = net.out_of_distribution_detection_score(data,
                                                                              metric)
    else:
        assert False
    if print_results:
        print('Uncertainty Benchmark "{}" of {} on {} with {} metric'.format(
            benchmark, net.name, type(data).__name__, metric))
        print('AUROC {:.3f}'.format(auroc))
        stdout.flush()
    return {'TPR': tpr, 'FPR': fpr, 'AUROC': auroc, 'thresholds': thresholds}


ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_mongo_observer())


@ex.command
def uncertainty_parameter_search(modelname, net_config, dataset, starting_weights,
                                 search_parameters, benchmark, uncertainty_metrics,
                                 _run):
    model = get_model(modelname)
    data = get_dataset(dataset['name'])

    def evaluation(parameters):
        with model(data_description=data.get_data_description(), **parameters) as net:
            measure_set = data(**dataset).get_measureset()
            import_weights_into_network(net, starting_weights)
            return {metric: evaluate_uncertainty(net, measure_set, metric,
                                                 benchmark=benchmark,
                                                 print_results=False)
                    for metric in uncertainty_metrics}
    _run.info['results'] = grid_search(evaluation, search_parameters, net_config)


@ex.automain
def uncertainty_benchmark(modelname, net_config, dataset, starting_weights, benchmark,
                          uncertainty_metrics, _run):
    model = get_model(modelname)
    data = get_dataset(dataset['name'])
    with model(data_description=data.get_data_description(), **net_config) as net:
        data = data(**dataset)
        import_weights_into_network(net, starting_weights)
        for metric in uncertainty_metrics:
            measurements = evaluate_uncertainty(net, data.get_testset(), metric,
                                                benchmark=benchmark)
            _run.info.setdefault('measurements', {})[metric] = measurements
