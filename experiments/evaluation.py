"""Evaluation of trained models."""
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from xview.datasets.synthia import AVAILABLE_SEQUENCES
from xview.models import get_model
from xview.settings import DATA_BASEPATH
from sys import stdout
from copy import deepcopy
import os

from .utils import ExperimentData, get_observer


def evaluate(net, data, print_results=True):
    """
    Evaluate the given network against the specified data and print the result.

    Args:
        net: An instance of a `base_model` class.
        data: A dataset
        print_results: If False, do not print measurements
    Returns:
        dict of measurements as produced by net.score, confusion matrix
    """
    # Load the dataset, we expect config to include the arguments
    measures, confusion_matrix = net.score(data.get_testset())

    if print_results:
        print('Evaluated network on %s:' % type(data).__name__)
        print('total accuracy {:.3f} mean F1 {:.3f} IoU {:.3f}'.format(
            measures['total_accuracy'], measures['mean_F1'], measures['mean_IoU']))
        for label in data.labelinfo:
            print("{:>15}: {:.2f} precision, {:.2f} recall, {:.2f} IoU".format(
                data.labelinfo[label]['name'], measures['precision'][label],
                measures['recall'][label], measures['IoU'][label]))

        # There seems to be a problem with capturing the print output, flush to be sure
        stdout.flush()
    return measures, confusion_matrix


def evaluate_on_all_synthia_seqs(net, data_config):
    """
    Evaluate a network on all synthia sequences individually.
    """
    adapted_config = deepcopy(data_config)
    all_measurements = {}
    for sequence in AVAILABLE_SEQUENCES:
        adapted_config['seqs'] = [sequence]
        measurements, _ = evaluate(net, adapted_config, print_results=False)
        print('Evaluated network on {}: {:.2f} IoU'.format(sequence,
                                                           measurements['mean_IoU']))
        all_measurements[sequence] = measurements
    stdout.flush()
    return all_measurements


def import_weights_into_network(net, starting_weights):
    """Based on either a list of descriptions of training experiments or one description,
    load the weights produced by these trainigns into the given network.

    Args:
        net: An instance of a `base_model` inheriting class.
        starting_weights: Either dict or list of dicts.
            if dict: expect key 'experiment_id' to match a previous experiment's ID.
                if key 'filename' is not set, will search for the first artifact that
                has 'weights' in the name.
            if list: a list of dicts where each dict will be evaluated as above
        kwargs are passed to net.import_weights
    """
    def import_weights_from_description(experiment_description, prefix=False):
        if experiment_description == 'paul_adapnet':
            net.import_weights(os.path.join(DATA_BASEPATH, 'Adapnet_weights_160000.npz'),
                               chill_mode=True, translate_prefix=prefix)
            return
        if experiment_description == 'imagenet_adapnet':
            net.import_weights(os.path.join(DATA_BASEPATH, 'resnet50_imagenet.npz'),
                               chill_mode=True, translate_prefix=prefix)
            return
        # description is an experiment id
        training_experiment = ExperimentData(experiment_description)
        net.import_weights(training_experiment.get_weights(), translate_prefix=prefix)

    if isinstance(starting_weights, list):
        for experiment_description in starting_weights:
            import_weights_from_description(experiment_description)
    elif isinstance(starting_weights, dict):
        for prefix, experiment_description in starting_weights.items():
            import_weights_from_description(experiment_description, prefix=prefix)
    else:
        import_weights_from_description(starting_weights)


ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())


@ex.command
def also_load_config(modelname, net_config, evaluation_data, starting_weights, _run):
    """In case of only a single training experiment, we also load the exact network
    config from this experiment as a default"""
    # Load the training experiment
    training_experiment = ExperimentData(starting_weights)

    model_config = training_experiment.get_record()['config']['net_config']
    model_config.update(net_config)
    model_config['gpu_fraction'] = 0.94

    # save this
    print('Running with net_config:')
    print(model_config)

    # Create the network
    model = get_model(modelname)
    with model(**model_config) as net:
        # import the weights
        import_weights_into_network(net, starting_weights)

        measurements, confusion_matrix = evaluate(net, evaluation_data)
        _run.info['measurements'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix


@ex.command
def all_synthia(modelname, net_config, evaluation_data, starting_weights, _run):
    """Load weigths from training experiments and evaluate network against specified
    data."""
    model = get_model(modelname)
    with model(**net_config) as net:
        import_weights_into_network(net, starting_weights)
        measurements = evaluate_on_all_synthia_seqs(net, evaluation_data)
        _run.info['measurements'] = measurements


@ex.main
def main(modelname, net_config, evaluation_data, starting_weights, _run):
    """Load weigths from training experiments and evaluate network against specified
    data."""
    model = get_model(modelname)
    with model(**net_config) as net:
        import_weights_into_network(net, starting_weights)
        measurements, confusion_matrix = evaluate(net, evaluation_data)
        _run.info['measurements'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix


if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
