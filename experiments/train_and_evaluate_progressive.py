from sacred import Experiment
from xview.models.progressive_fcn import ProgressiveFCN
from xview.datasets.synthia import AVAILABLE_SEQUENCES
import numpy as np
from copy import deepcopy

from experiments.utils import get_mongo_observer, ExperimentData
from experiments.training import create_directories, train_network, load_data
from experiments.evaluation import evaluate, evaluate_on_all_synthia_seqs


ex = Experiment()
ex.observers.append(get_mongo_observer())


@ex.command
def rgb_to_depth(net_config, data_config, starting_weights, num_iterations, _run):
    """Training for progressive FCN with transfer from existing RGB column to Depth."""
    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # Get data from all synthia sequences
    all_sequences = {}
    if data_config['dataset'] == 'synthia':
        adapted_config = deepcopy(data_config)
        for sequence in AVAILABLE_SEQUENCES:
            adapted_config['seqs'] = [sequence]
            data = load_data(adapted_config)
            all_sequences[sequence] = data.get_validation_data(num_items=10)

    # Get the existing RGB weights
    training_experiment = ExperimentData(starting_weights['experiment_id'])
    # Take the first weights file there is in this experiment
    filename = (artifact['name']
                for artifact in training_experiment.get_record()['artifacts']
                if 'weights' in artifact['name']).next()
    rgb_weights = np.load(training_experiment.get_artifact(filename))

    # For the first layer, take the mean over all 3 channels
    # Therefore, we have to define a new weights dict.
    new_weights = {key: rgb_weights[key] for key in rgb_weights}
    new_weights['rgb_conv1_1/kernel'] = rgb_weights['rgb_conv1_1/kernel'].mean(3, keepdims=True)

    # We need a file handler for this new weights dict, therefore we save the weights
    # into a temporary file.
    np.savez('/tmp/translated_rgb_weights.npz', **new_weights)

    # create the network
    with ProgressiveFCN(output_dir=output_dir, **net_config) as net:
        # import the above created weights
        net.import_weights('/tmp/translated_rgb_weights.npz')

        train_network(net, output_dir, data_config, num_iterations,
                      starting_weights=False, experiment=ex,
                      additional_eval_data=all_sequences)

        print('INFO Evaluate the network against the training sequences')
        evaluate(net, data_config)

        print('INFO: Evaluating against all sequences')
        evaluate_on_all_synthia_seqs(net, data_config)


@ex.automain
def main(net_config, data_config, starting_weights, num_iterations, _run):
    """Training for progressive FCN."""
    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # create the network
    with ProgressiveFCN(output_dir=output_dir, **net_config) as net:
        train_network(net, output_dir, data_config, num_iterations, starting_weights,
                      experiment=ex)

        print('INFO Evaluate the network adainst the training sequences')
        evaluate(net, data_config)

        print('INFO: Evaluating against all sequences')
        evaluate_on_all_synthia_seqs(net, data_config)
