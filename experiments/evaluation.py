from sacred import Experiment
from experiments.utils import ExperimentData, get_mongo_observer
from xview.datasets import get_dataset
from xview.models import get_model
import numpy as np
import os
import shutil
from sys import stdout


ex = Experiment()
ex.observers.append(get_mongo_observer())


def evaluate(net, data_config):
    """Evaluate the given network against the specified data and report the result
    to the given experiment."""
    # Load the dataset, we expect config to include the arguments
    dataset = get_dataset(data_config['dataset'])
    data = dataset(data_config['sequences'], 1)
    if data_config.get('use_trainset', False):
        print('INFO: Evaluating against trainset')
        batches = data.get_train_data(batch_size=1)
    else:
        batches = data.get_test_data(batch_size=1)

    measures, confusion_matrix = net.score(batches)

    print('Evaluated network on {}:'.format(data_config['dataset']))
    print('total accuracy {:.2f} mean F1 {:.2f} IU {:.2f}'.format(
        measures['total_accuracy'], measures['mean_F1'], measures['mean_IU']))
    for label in data.labelinfo:
        print("{:>15}: {:.2f} precision, {:.2f} recall, {:.2f} F1".format(
            data.labelinfo[label]['name'], measures['precision'][label],
            measures['recall'][label], measures['F1'][label]))

    # There seems to be a problem with capturing the print output, flush it for security.
    stdout.flush()
    return measures, confusion_matrix


def import_weights_into_network(net, starting_weights, **kwargs):
    """Based on either a list of descriptions of training experiments or one description,
    load the weights produced by these trainigns into the given network.
    kwargs are passed to net.import_weights
    """
    def import_weights_from_description(experiment_description):
        training_experiment = ExperimentData(experiment_description['experiment_id'])
        if 'filename' not in experiment_description:
            # If no specific file specified, take first found
            filename = (artifact['name']
                        for artifact in training_experiment.get_record()['artifacts']
                        if 'weights' in artifact['name']).next()
        else:
            filename = experiment_description['filename']
        net.import_weights(training_experiment.get_artifact(filename), **kwargs)

    if isinstance(starting_weights, list):
        for experiment_description in starting_weights:
            import_weights_from_description(experiment_description)
    else:
        import_weights_from_description(starting_weights)


@ex.command
def multiple_weights(data_config, modelname, net_config, starting_weights, _run):
    """Special command in case we want to import weights from multiple experiments after
    each other"""
    train_ids = []

    model = get_model(modelname)
    with model(**net_config) as net:
        for weights_descriptor in starting_weights:
            training_experiment = ExperimentData(weights_descriptor['experiment_id'])
            train_ids.append(weights_descriptor['experiment_id'])
            weights = training_experiment.get_artifact(weights_descriptor['filename'])
            net.import_weights(weights)

        evaluate(net, data_config)

    # save the reference to the experiments
    _run.info['training_ids'] = train_ids


@ex.automain
def my_main(data_config, modelname, net_config, starting_weights, _run):
    """Standard command"""
    # This is an independent experiment, but still we want to set a connection to the
    # training run.
    _run.info['training_id'] = starting_weights['experiment_id']
    # Load the training experiment
    training_experiment = ExperimentData(starting_weights['experiment_id'])
    # We take the network configuration from this experiment as a basis and overwrite
    # any given new values.
    model_config = training_experiment.get_record()['config']['fcn_config']
    model_config.update(net_config)

    # save this
    _run.config['net_config'] = model_config
    print('Running with net_config:')
    print(model_config)

    # Create the network
    model = get_model(modelname)
    with model(**model_config) as net:
        # import the weights
        import_weights_into_network(net, starting_weights)

        evaluate(net, data_config)
