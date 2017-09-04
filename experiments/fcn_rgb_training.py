from sacred import Experiment
from sacred.utils import TimeoutInterrupt
from xview.datasets.synthia import Synthia
from xview.models.simple_fcn import SimpleFCN
from xview.settings import DATA_BASEPATH
import os
import shutil

from experiments.utils import get_mongo_observer, ExperimentData


ex = Experiment()
ex.observers.append(get_mongo_observer())

@ex.automain
def my_main(data_config, fcn_config, num_iterations, starting_weights, _run):
    # create temporary directory for output files
    if os.path.exists('/tmp/fcn_train'):
        # Remove old data
        shutil.rmtree('/tmp/fcn_train')
    os.mkdir('/tmp/fcn_train')
    # The id of this experiment is stored in the magical _run object we get from the
    # decorator.
    output_dir = '/tmp/fcn_train/{}'.format(_run._id)
    os.mkdir(output_dir)

    # Tell the experiment that this output dir is also used for tensorflow summaries
    ex.info.setdefault("tensorflow", {}).setdefault("logdirs", []).append(output_dir)

    # Load the dataset, we expect config to include the arguments
    data = Synthia(data_config['sequences'], data_config['batchsize'])
    # get validation set
    validation_set = data.get_validation_data(num_items=10)

    # create the network
    with SimpleFCN(output_dir=output_dir, **fcn_config) as net:

        if starting_weights:
            # load startign weights
            if starting_weights == 'washington':
                # load the washington weights
                weights = os.path.join(DATA_BASEPATH, 'darnn/FCN_weights_40000.npz')
            elif isinstance(starting_weights, dict):
                print('INFO: Loading weights from experiment {}'.format(
                    starting_weights['experiment_id']))
                # load weights from previous experiment
                previous_exp = ExperimentData(starting_weights['experiment_id'])
                weights = previous_exp.get_artifact(starting_weights['filename'])
            net.import_weights(weights, chill_mode=True)

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
