from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.utils import get_mongo_observer
from experiments.evaluation import import_weights_into_network
from experiments.utils import ExperimentData
from xview.datasets import get_dataset
from xview.models import get_model, BayesMix, AverageMix
from copy import deepcopy
from sys import stdout
from sklearn.model_selection import train_test_split
import numpy as np
from os import path, mkdir


ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_mongo_observer())


def split_test_data(data_config):
    # Load the dataset, we expect config to include the arguments
    dataset_params = {key: val for key, val in data_config.items()
                      if key not in ['dataset']}
    dataset_params['augmentation'] = {
        key: False for key in ['crop', 'scale', 'vflip', 'hflip', 'gamma', 'rotate',
                               'shear', 'contrast', 'brightness']}
    data = get_dataset(data_config['dataset'], dataset_params)

    measure_set, test_set = train_test_split(data.testset, test_size=.5, random_state=1)

    return data, measure_set, test_set


@ex.command
def collect_data(fitting_experiment):
    exp = ExperimentData(fitting_experiment)
    evaluation_data = exp.get_record()['config']['evaluation_data']
    net_config = exp.get_record()['config']['net_config']
    starting_weights = exp.get_record()['config']['starting_weights']
    confusion_matrices = exp.get_record()['info']['confusion_matrices']
    # load numpy confusion matrices
    confusion_matrices = {
        key: np.array(val['values']) for key, val in confusion_matrices.items()
    }

    data, _, _ = split_test_data(evaluation_data)

    # now collect insight on bayes mix
    predictions = []
    likelihoods = []
    conditionals = []
    probs = []
    with BayesMix(confusion_matrices=confusion_matrices, **net_config) as net:
        import_weights_into_network(net, starting_weights)
        for batch in data.get_test_data():
            insight = net.get_insight(batch)
            probs.append(insight[0])
            likelihoods.append(insight[1])
            conditionals.append(insight[2])
            predictions.append(insight[3])

    outpath = '/cluster/work/riner/users/blumh/measurements/{}'.format(
        fitting_experiment)
    mkdir(outpath)
    np.savez(path.join(outpath, 'predictions.npz'), *predictions)
    np.savez(path.join(outpath, 'likelihoods.npz'), *likelihoods)
    np.savez(path.join(outpath, 'conditionals.npz'), *conditionals)
    np.savez(path.join(outpath, 'probs.npz'), *probs)


@ex.command
def evaluate(net_config, evaluation_data, modelname, starting_weights, _run):
    """Load weigths from training experiments and evalaute fusion against specified
    data."""
    data, _, test_set = split_test_data(evaluation_data)

    model = get_model(modelname)
    # now evaluate average mix
    with model(**net_config) as net:
        import_weights_into_network(net, starting_weights)
        measurements, confusion_matrix = net.score(data.get_set_data(test_set))
        _run.info['measurements'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix

    print('Evaluated on {} data:'.format(evaluation_data['dataset']))
    print('total accuracy {:.3f} IoU {:.3f}'.format(measurements['total_accuracy'],
                                                    measurements['mean_IoU']))

    # There seems to be a problem with capturing the print output, flush to be sure
    stdout.flush()


@ex.command
def average(net_config, evaluation_data, starting_weights, _run):
    """Load weigths from training experiments and evalaute fusion against specified
    data."""
    data, _, _ = split_test_data(evaluation_data)

    # now evaluate average mix
    with AverageMix(**net_config) as net:
        import_weights_into_network(net, starting_weights)
        measurements, confusion_matrix = net.score(data.get_test_data())
        _run.info['measurements'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix

    print('Evaluated Average Fusion on {} data:'.format(evaluation_data['dataset']))
    print('total accuracy {:.3f} IoU {:.3f}'.format(measurements['total_accuracy'],
                                                    measurements['mean_IoU']))

    # There seems to be a problem with capturing the print output, flush to be sure
    stdout.flush()


@ex.automain
def fit_and_evaluate(net_config, evaluation_data, starting_weights, _run):
    """Load weigths from training experiments and evalaute fusion against specified
    data."""
    data, _, _ = split_test_data(evaluation_data)

    # evaluate individual experts
    model = get_model(net_config['expert_model'])
    confusion_matrices = {}
    for expert in net_config['num_channels']:
        model_config = deepcopy(net_config)
        model_config['num_channels'] = net_config['num_channels'][expert]
        model_config['modality'] = expert
        model_config['prefix'] = net_config['prefixes'][expert]
        with model(**model_config) as net:
            import_weights_into_network(net, starting_weights[model_config['prefix']])
            m, conf_mat = net.score(data.get_measure_data())
            confusion_matrices[expert] = conf_mat
            print('Evaluated network {} on {} train data:'.format(
                expert, evaluation_data['dataset']))
            print("INFO now getting test results")
            m, _ = net.score(data.get_test_data())
            print('total accuracy {:.3f} IoU {:.3f}'.format(m['total_accuracy'],
                                                            m['mean_IoU']))
        _run.info.setdefault('measurements', {}).setdefault(expert, m)
    _run.info['confusion_matrices'] = confusion_matrices

    # now evaluate bayes mix
    with BayesMix(confusion_matrices=confusion_matrices, **net_config) as net:
        import_weights_into_network(net, starting_weights)
        measurements, confusion_matrix = net.score(data.get_test_data())
        _run.info['measurements']['fusion'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix

    print('Evaluated Bayes Fusion on {} data:'.format(evaluation_data['dataset']))
    print('total accuracy {:.3f} IoU {:.3f}'.format(measurements['total_accuracy'],
                                                    measurements['mean_IoU']))

    # There seems to be a problem with capturing the print output, flush to be sure
    stdout.flush()
