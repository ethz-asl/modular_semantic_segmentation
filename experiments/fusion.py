from sacred import Experiment
from experiments.utils import ExperimentData, get_mongo_observer
from experiments.evaluation import evaluate, import_weights_into_network
from xview.datasets import get_dataset
from xview.models.mix_fcn import MixFCN
from sys import stdout
import os


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


@ex.automain
def fit_and_evaluate(net_config, evaluation_data, starting_weights, _run):
    """Load weigths from trainign experiments and evalaute network against specified
    data."""
    output_dir = '/tmp/mix_fcn'

    with MixFCN(output_dir=output_dir, **net_config) as net:
        import_weights_into_network(net, starting_weights)

        # Measure the single experts against the trainingset.
        dataset_params = {key: val for key, val in evaluation_data.items()
                          if key not in ['dataset', 'use_trainset']}
        dataset_params['batchsize'] = 1
        # Load the dataset, we expect config to include the arguments
        data = get_dataset(evaluation_data['dataset'], dataset_params)
        batches = data.get_train_data(batch_size=10)
        net.fit(batches)

        measurements, confusion_matrix = evaluate(net, evaluation_data)
        _run.info['measurements'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix

        # To end the experiment, we collect all produced output files and store them.
        for filename in os.listdir(output_dir):
            ex.add_artifact(os.path.join(output_dir, filename))
