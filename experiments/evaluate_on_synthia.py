from sacred import Experiment
from experiments.utils import ExperimentData, get_mongo_observer
from xview.datasets.synthia import Synthia
from xview.models import get_model
import numpy as np
import os
import shutil


ex = Experiment()
ex.observers.append(get_mongo_observer())


def evaluate(net, sequences):
    # Load the dataset, we expect config to include the arguments
    data = Synthia(sequences, 1)
    testdata = data.get_test_data(batch_size=5)

    measures, confusion_matrix = net.score(testdata)

    print('Evaluated network on Synthia:')
    print('MEAN accuracy {:.2f} IU {:.2f}'.format(measures['mean_accuracy'],
                                                  measures['mean_IU']))
    for label in data.labelinfo:
        print("{:>15}: {:.2f} accuracy".format(data.labelinfo[label]['name'],
                                               measures['accuracy'][label]))

    # create temporary directory for output files
    if os.path.exists('/tmp/eval'):
        # Remove old data
        shutil.rmtree('/tmp/eval')
    os.mkdir('/tmp/eval')

    # As this evaluation will take quite some time, we store the confusion matrix for
    # possible later use.
    np.save('/tmp/eval/confusion_matrix.npy', confusion_matrix)
    ex.add_artifact('/tmp/eval/confusion_matrix.npy')


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
        weights = training_experiment.get_artifact(starting_weights['filename'])
        net.import_weights(weights)

        evaluate(net, data_config['sequences'])


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

        evaluate(net, data_config['sequences'])

    # save the reference to the experiments
    _run.info['training_ids'] = train_ids
