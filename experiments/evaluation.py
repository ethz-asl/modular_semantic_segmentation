"""Evaluation of trained models."""
from sacred import Experiment
from experiments.utils import ExperimentData, get_mongo_observer
from xview.datasets import get_dataset
from xview.models import get_model
from sys import stdout


def evaluate(net, data_config):
    """Evaluate the given network against the specified data and report the result
    to the given experiment."""
    # Load the dataset, we expect config to include the arguments
    dataset = get_dataset(data_config['dataset'])
    data = dataset(data_config['sequences'], 1)
    if data_config.get('use_trainset', default=False):
        print('INFO: Evaluating against trainset')
        batches = data.get_train_data(batch_size=1)
    else:
        batches = data.get_test_data(batch_size=1)

    measures, confusion_matrix = net.score(batches)

    print('Evaluated network on {}:'.format(data_config['dataset']))
    print('total accuracy {:.2f} mean F1 {:.2f} IoU {:.2f}'.format(
        measures['total_accuracy'], measures['mean_F1'], measures['mean_IoU']))
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


ex = Experiment()
ex.observers.append(get_mongo_observer())


@ex.config_hook
def load_model_configuration(config, command_name, logger):

    def get_config_for_experiment(id):
        training_experiment = ExperimentData(id)
        return training_experiment.get_record()['config']

    # This hook will produce the following update-dict for the config:
    cfg_update = {}

    if isinstance(config['starting_weights'], list):
        # For convenience, we simply record all the configurations of the trainign
        # experiments.
        cfg_update['starting_weights'] = []
        for exp_descriptor in config['starting_weights']:
            cfg_update['starting_weights'].append({'config': get_config_for_experiment(
                exp_descriptor['experiment_id'])})
    else:
        train_exp_config = get_config_for_experiment(
            config['starting_weights']['experiment_id'])
        # First, same as above, capture the information
        cfg_update['starting_weights'] = {'config': train_exp_config}
    return cfg_update


@ex.command
def also_load_config(modelname, net_config, evaluation_data, starting_weights, _run):
    """In case of only a single training experiment, we also load the exact network
    config from this experiment as a default"""
    # Load the training experiment
    training_experiment = ExperimentData(starting_weights['experiment_id'])

    model_config = training_experiment.get_record()['config']['fcn_config']
    model_config.update(net_config)

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


@ex.automain
def main(modelname, net_config, evaluation_data, starting_weights, _run):
    """Load weigths from training experiments and evaluate network against specified
    data."""
    model = get_model(modelname)
    with model(**net_config) as net:
        import_weights_into_network(net, starting_weights)
        measurements, confusion_matrix = evaluate(net, evaluation_data)
        _run.info['measurements'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix
