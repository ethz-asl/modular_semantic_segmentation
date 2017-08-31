from sacred import Experiment
from xview.datasets.synthia import Synthia
from xview.models.simple_fcn import SimpleFCN
from xview.settings import DATA_BASEPATH
import os
import shutil

from experiments.utils import get_mongo_observer


ex = Experiment()
ex.observers.append(get_mongo_observer())


@ex.automain
def my_main(data_config, fcn_config, num_iterations):
    # create temporary directory for output files
    output_dir = '/tmp/fcn_train'
    if os.path.exists(output_dir):
        # Remove old data
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # Load the dataset, we expect config to include the arguments
    data = Synthia(data_config['sequences'], data_config['batchsize'])
    # get validation set
    validation_set = data.get_test_data(num_items=20)

    # create the network
    with SimpleFCN(output_dir=output_dir, **fcn_config) as net:

        # load the washington weights
        washington_weights = os.path.join(DATA_BASEPATH,
                                          'darnn/FCN_weights_40000.npz')
        net.import_weights(washington_weights, chill_mode=True)
        # We also tell the experiment that we used this resource
        ex.add_resource(washington_weights)

        # finetune on synthia
        net.fit(data, num_iterations, validation_data=validation_set)

        # Store the weights into the standard output directory
        net.export_weights()

    # To end the experiment, we collect all produced output files and store them.
    for filename in os.listdir(output_dir):
        ex.add_artifact(os.path.join(output_dir, filename))
