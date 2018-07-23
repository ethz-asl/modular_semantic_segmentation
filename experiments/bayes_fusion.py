from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.utils import get_observer
from experiments.evaluation import import_weights_into_network
from experiments.utils import ExperimentData
from xview.datasets import get_dataset
from xview.models import get_model, BayesFusion, AverageFusion
from copy import deepcopy
from sys import stdout
from sklearn.model_selection import train_test_split
import numpy as np
from os import path, mkdir


ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())


def split_test_data(data_config):
    # Load the dataset, we expect config to include the arguments
    dataset_params = {key: val for key, val in data_config.items()
                      if key not in ['dataset']}
    dataset_params['augmentation'] = {
        key: False for key in ['crop', 'scale', 'vflip', 'hflip', 'gamma', 'rotate',
                               'shear', 'contrast', 'brightness']}
    data = get_dataset(data_config['dataset'])(**dataset_params)

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
    with BayesFusion(confusion_matrices=confusion_matrices, **net_config) as net:
        import_weights_into_network(net, starting_weights)
        for batch in data.get_test_data():
            insight = net.get_insight(batch)
            probs.append(insight[0])
            likelihoods.append(insight[1])
            conditionals.append(insight[2])
            predictions.append(insight[3])

    outpath = '/cluster/work/riner/users/blumh/measurements/{}'.format(
        fitting_experiment)
    if not path.exists(outpath):
        mkdir(outpath)
    np.savez_compressed(path.join(outpath, 'predictions.npz'), *predictions)
    np.savez_compressed(path.join(outpath, 'likelihoods.npz'), *likelihoods)
    np.savez_compressed(path.join(outpath, 'conditionals.npz'), *conditionals)
    np.savez_compressed(path.join(outpath, 'probs.npz'), *probs)


@ex.command
def evaluate(net_config, evaluation_data, modelname, starting_weights, _run):
    """Load weigths from training experiments and evalaute fusion against specified
    data."""
    data = get_dataset(evaluation_data['dataset'])

    model = get_model(modelname)
    # now evaluate average mix
    with model(data_description=data.get_data_description(), **net_config) as net:
        data = data(**evaluation_data)
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
    data = get_dataset(evaluation_data['dataset'])

    # now evaluate average mix
    with AverageFusion(data_description=data.get_data_description(), **net_config) as net:
        data = data(**evaluation_data)
        import_weights_into_network(net, starting_weights)
        measurements, confusion_matrix = net.score(data.get_testset())
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
    dataset = get_dataset(evaluation_data['dataset'])

    # evaluate individual experts
    model = get_model(net_config['expert_model'])
    confusion_matrices = {}
    for expert in net_config['num_channels']:
        model_config = deepcopy(net_config)
        model_config['modality'] = expert
        model_config['prefix'] = net_config['prefixes'][expert]
        with model(data_description=dataset.get_data_description(), **model_config) as net:
            data = dataset(**evaluation_data)
            import_weights_into_network(net, starting_weights[model_config['prefix']])
            m, conf_mat = net.score(data.get_measureset())
            confusion_matrices[expert] = conf_mat
            print('Evaluated network {} on {} measurement set:'.format(
                expert, evaluation_data['dataset']))
            print("INFO now getting test results")
            m, _ = net.score(data.get_testset())
            print('total accuracy {:.3f} IoU {:.3f}'.format(m['total_accuracy'],
                                                            m['mean_IoU']))
        _run.info.setdefault('measurements', {}).setdefault(expert, m)
    _run.info['confusion_matrices'] = confusion_matrices

    # now evaluate bayes mix
    with BayesFusion(data_description=dataset.get_data_description(),
                     confusion_matrices=confusion_matrices, **net_config) as net:
        data = dataset(**evaluation_data)
        import_weights_into_network(net, starting_weights)
        measurements, confusion_matrix = net.score(data.get_testset())
        _run.info['measurements']['fusion'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix

    print('Evaluated Bayes Fusion on {} data:'.format(evaluation_data['dataset']))
    print('total accuracy {:.3f} IoU {:.3f}'.format(measurements['total_accuracy'],
                                                    measurements['mean_IoU']))

    # There seems to be a problem with capturing the print output, flush to be sure
    stdout.flush()
