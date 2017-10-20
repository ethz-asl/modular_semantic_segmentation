from sacred import Experiment
from sacred.utils import TimeoutInterrupt
from xview.datasets.synthia import Synthia
from xview.models.simple_fcn import SimpleFCN
from xview.models.progressive_fcn import ProgressiveFCN
from xview.settings import DATA_BASEPATH
import os
import shutil

from experiments.utils import get_mongo_observer, ExperimentData


def create_directories(run_id, experiment):
    # create temporary directory for output files
    if os.path.exists('/tmp/fcn_train'):
        # Remove old data
        shutil.rmtree('/tmp/fcn_train')
    os.mkdir('/tmp/fcn_train')
    # The id of this experiment is stored in the magical _run object we get from the
    # decorator.
    output_dir = '/tmp/fcn_train/{}'.format(run_id)
    os.mkdir(output_dir)

    # Tell the experiment that this output dir is also used for tensorflow summaries
    experiment.info.setdefault("tensorflow", {}).setdefault("logdirs", [])\
        .append(output_dir)
    return output_dir


def import_startingweights(net, starting_weights):
    # load startign weights
    if starting_weights == 'washington':
        # load the washington weights
        weights = os.path.join(DATA_BASEPATH, 'darnn/FCN_weights_40000.npz')
        net.import_weights(weights, chill_mode=True)
    elif isinstance(starting_weights, dict):
        print('INFO: Loading weights from experiment {}'.format(
            starting_weights['experiment_id']))
        # load weights from previous experiment
        previous_exp = ExperimentData(starting_weights['experiment_id'])
        weights = previous_exp.get_artifact(starting_weights['filename'])
        net.import_weights(weights, chill_mode=True)
    elif isinstance(starting_weights, list):
        for weights in starting_weights:
            import_startingweights(net, weights)


ex = Experiment()
ex.observers.append(get_mongo_observer())


@ex.capture
def train_network(net, output_dir, data_config, num_iterations, starting_weights):
    # Load the dataset, we expect config to include the arguments
    data = Synthia(data_config['sequences'], data_config['batchsize'],
                   direction=data_config.get('direction', 'F'))
    # get validation set
    validation_set = data.get_validation_data(num_items=6)

    # Train the given network
    if starting_weights:
        import_startingweights(net, starting_weights)

    try:
        # finetune on synthia
        net.fit(data, num_iterations, validation_data=validation_set)
        timeout = False
    except KeyboardInterrupt:
        print('WARNING: Got Keyboard Interrupt, will save weights and close')
        timeout = True

    # Store the weights into the standard output directory
    net.export_weights()

    # To end the experiment, we collect all produced output files and store them.
    for filename in os.listdir(output_dir):
        ex.add_artifact(os.path.join(output_dir, filename))

    if timeout:
        raise TimeoutInterrupt


@ex.command
def progressive(fcn_config, _run):
    """Training for progressive FCN."""
    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # create the network
    with ProgressiveFCN(output_dir=output_dir, **fcn_config) as net:
        train_network(net, output_dir)


@ex.automain
def my_main(fcn_config, _run):
    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # create the network
    with SimpleFCN(output_dir=output_dir, **fcn_config) as net:
        train_network(net, output_dir)
