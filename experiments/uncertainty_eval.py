"""Evaluation of trained uncertainty models."""
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers.mongo import force_bson_encodeable
from xview.models import get_model
from xview.datasets import get_dataset
from sys import stdout
import numpy as np
import threading
import os

from .utils import get_mongo_observer, ExperimentData
from .evaluation import import_weights_into_network
from .different_evaluation_parameters import grid_search
from .training import create_directories, train_network


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


def measure_metrics(net, data, metrics, label_flip=None):
    nll, class_count = net.nll_score(data)
    ret = {'nll': nll, 'class_counts': class_count}
    for metric in metrics:
        ret[metric] = net.value_distribution(data, metric)
    if label_flip:
        # measure the difference to the distribution we would expect based on the
        # label_flip augmentation during the training
        prior = np.zeros(net.config['num_classes'])
        prior[label_flip[0]] = 1 - label_flip[2]
        prior[label_flip[1]] = label_flip[2]
        ret['distribution_miscalibration'] = net.mean_diff(
            data, prior, condition=lambda t, c: np.logical_or(c == label_flip[0],
                                                              c == label_flip[1]))
    dirichlet, mean = net.prob_distribution(data)
    ret['dirichlet_priors'] = dirichlet
    ret['mean_output'] = mean
    return ret


ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_mongo_observer())


@ex.command
def evaluate_temperature_scaling(experiments, temperatures):
    for exp_id in experiments:
        exp = ExperimentData(exp_id)
        dataset = exp.get_record()['info']['dataset']
        data = get_dataset(dataset['name'])
        data_description = list(data.get_data_description())
        if exp.get_record()['config']['method'] == 'new_class':
            data_description[2] += 1
        model = get_model(exp.get_record()['config']['modelname'])

        # get the label_flip specs
        label_flip = None
        if 'label_flip' in dataset['augmentation']:
            label_flip = dataset['augmentation']['label_flip']

        def evaluation(parameters):
            with model(data_description=data_description, **parameters) as net:
                import_weights_into_network(net, exp_id)
                return measure_metrics(net, data(**dataset).get_testset(),
                                       exp.get_record()['config']['uncertainty_metrics'],
                                       label_flip=label_flip)

        result = grid_search(evaluation, {'temperature_scaling': temperatures},
                             exp.get_record()['config']['net_config'])
        info = exp.get_record()['info']
        info['temperature_grid_search'] = force_bson_encodeable(result)
        exp.update_record({'info': info})


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


@ex.command
def train_ambiguous(modelname, net_config, dataset, starting_weights, method,
                    num_iterations, uncertainty_metrics, _run):
    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    data = get_dataset(dataset['name'])
    data_description = list(data.get_data_description())
    num_classes = data_description[2]

    args = False
    if isinstance(method, list):
        args = method[1:]
        method = method[0]
    # augmentate the class labels
    if method == 'flip_classes':
        # randomly map two classes onto each other to make them ambiguos
        classes = np.random.choice(list(range(num_classes)), size=2, replace=False)
        dataset.setdefault('augmentation', {})['label_flip'] = (classes[0], classes[1],
                                                                np.random.rand())
    elif method == 'new_class':
        # randomly label a given class as a new, nonexisting class
        data_description[2] = num_classes + 1
        if args:
            old_class = args[0]
            print(args)
        else:
            old_class = np.random.choice(list(range(num_classes)))
        dataset.setdefault('augmentation', {})['label_flip'] = (old_class, num_classes,
                                                                np.random.rand())
    elif method == 'merge':
        # randomly merge two classes together
        classes = np.random.choice(list(range(num_classes)), size=2, replace=False)
        dataset.setdefault('augmentation', {})['label_merge'] = (classes[0], classes[1])
    _run.info.setdefault('dataset', {}).update(dataset)

    model = get_model(modelname)
    with model(data_description=data_description, output_dir=output_dir,
               **net_config) as net:
        data = data(**dataset)
        train_network(net, output_dir, data, num_iterations, starting_weights, ex)
        _run.info['measurements'] = measure_metrics(
            net, data.get_testset(), uncertainty_metrics,
            label_flip=dataset['augmentation'].get('label_flip', None))
        net.close()
    print(threading.enumerate())


@ex.command
def measure(modelname, net_config, dataset, starting_weights, uncertainty_metrics, _run):
    model = get_model(modelname)
    data = get_dataset(dataset['name'])
    data_description = list(data.get_data_description())
    if 'num_classes' in dataset:
        data_description[2] = dataset['num_classes']
    with model(data_description=data_description, **net_config) as net:
        data = data(**dataset)
        import_weights_into_network(net, starting_weights)
        _run.info['measurements'] = measure_metrics(net, data.get_testset(),
                                                    uncertainty_metrics)


@ex.main
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


if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
