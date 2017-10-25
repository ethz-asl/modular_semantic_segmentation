from sacred import Experiment
from xview.models.progressive_fcn import ProgressiveFCN

from experiments.utils import get_mongo_observer
from experiments.fcn_train import create_directories, train_network
from experiments.evaluation import evaluate


ex = Experiment()
ex.observers.append(get_mongo_observer())


@ex.capture
def captured_train_network(net, output_dir, data_config, num_iterations,
                           starting_weights):
    train_network(net, output_dir, data_config, num_iterations, starting_weights)


@ex.automain
def my_main(fcn_config, data_config, evaluation_data, _run):
    """Training for progressive FCN."""
    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # create the network
    with ProgressiveFCN(output_dir=output_dir, **fcn_config) as net:
        captured_train_network(net, output_dir)

        print('INFO Evaluate the network adainst the training sequences')
        evaluate(net, data_config)

        print('INFO: Evaluating against seperate data')
        evaluate(net, evaluation_data)
