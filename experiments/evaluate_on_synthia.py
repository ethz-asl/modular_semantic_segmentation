from sacred import Experiment
from experiments.utils import ExperimentData, get_mongo_observer
from xview.datasets.synthia import Synthia
from xview.models import get_model
import numpy as np
import os
import shutil

ex = Experiment()
ex.observers.append(get_mongo_observer())

@ex.automain
def my_main(data_config, modelname, net_config, starting_weights, _run):
    # create temporary directory for output files
    if os.path.exists('/tmp/eval'):
        # Remove old data
        shutil.rmtree('/tmp/eval')
    os.mkdir('/tmp/eval')

    # Load the dataset, we expect config to include the arguments
    data = Synthia(data_config['sequences'], 1)
    testdata = data.get_test_data(batch_size=5)

    # This is an independent experiment, but still we want to set a connection to the
    # training run.
    _run.info['training_id'] = starting_weights['experiment_id']

    # Create the network
    model = get_model(modelname)
    with model(**net_config) as net:
        # import the weights
        training_experiment = ExperimentData(starting_weights['experiment_id'])
        weights = training_experiment.get_artifact(starting_weights['filename'])
        net.import_weights(weights)

        measures, confusion_matrix = net.score(testdata)

    print('Evaluated network on Synthia:')
    print('MEAN accuracy {:.2f} IU {:.2f}'.format(measures['mean_accuracy'],
                                                  measures['mean_IU']))
    for label in data.labelinfo:
        print("{:>15}: {:.2f} accuracy".format(data.labelinfo[label]['name'],
                                               measures['accuracy'][label]))

    # As this evaluation will take quite some time, we store the confusion matrix for
    # possible later use.
    np.save('/tmp/eval/confusion_matrix.npz', confusion_matrix)
    ex.add_artifact('/tmp/eval/confusion_matrix.npz')
