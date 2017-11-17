from sacred import Experiment
from xview.models.simple_fcn import SimpleFCN
import numpy as np

from experiments.utils import ExperimentData, get_mongo_observer
from experiments.fcn_training import create_directories, train_network
from experiments.evaluation import evaluate

ex = Experiment()
ex.observers.append(get_mongo_observer())


@ex.automain
def rgb_to_depth(net_config, data_config, num_iterations, starting_weights, 
                 evaluation_data, _run):
    """Training for progressive FCN."""
    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

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
    new_weights['rgb_conv1_1/kernel'] = rgb_weights['rgb_conv1_1/kernel'].mean(2, keepdims=True)

    # We need a file handler for this new weights dict, therefore we save the weights
    # into a temporary file.
    np.savez('/tmp/translated_rgb_weights.npz', **new_weights)

    # create the network
    with SimpleFCN(output_dir=output_dir, **net_config) as net:
        # import the above created weights
        net.import_weights('/tmp/translated_rgb_weights.npz')

        train_network(net, output_dir, data_config, num_iterations,
                      starting_weights=False, experiment=ex)

        print('INFO: Evaluate the network adainst the training sequences')
        evaluate(net, data_config)

        print('INFO: Evaluating against seperate data')
        evaluate(net, evaluation_data)
