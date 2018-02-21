from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.utils import get_mongo_observer
from experiments.evaluation import evaluate, import_weights_into_network
from xview.datasets import get_dataset
from xview.models import get_model, BayesMix
from copy import deepcopy
from sys import stdout


ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_mongo_observer())

@ex.automain
def fit_and_evaluate(net_config, evaluation_data, starting_weights, _run):
    """Load weigths from training experiments and evalaute fusion against specified
    data."""

    # Load the dataset, we expect config to include the arguments
    dataset_params = {key: val for key, val in evaluation_data.items()
                      if key not in ['dataset', 'use_trainset']}
    dataset_params['augmentation'] = {
        key: False for key in ['crop', 'scale', 'vflip', 'hflip', 'gamma', 'rotate',
                               'shear', 'contrast', 'brightness']}
    data = get_dataset(evaluation_data['dataset'], dataset_params)

    # evaluate individual experts
    model = get_model(net_config['expert_model'])
    confusion_matrices = {}
    for expert in net_config['num_channels']:
        model_config = deepcopy(net_config)
        model_config['num_channels'] = net_config['num_channels'][expert]
        model_config['modality'] = expert
        model_config['prefix'] = net_config['prefixes'][expert]
        with model(**model_config) as net:
            import_weights_into_network(net, starting_weights[model_config['prefix']])
            if evaluation_data['use_trainset']:
                m, conf_mat = net.score(data.get_train_data(training_format=False))
            else:
                m, conf_mat = net.score(data.get_validation_data()())
            confusion_matrices[expert] = conf_mat
            print('Evaluated network {} on {} train data:'.format(
                expert, evaluation_data['dataset']))
            print('total accuracy {:.3f} IoU {:.3f}'.format(m['total_accuracy'],
                                                            m['mean_IoU']))
    _run.info['confusion_matrices'] = confusion_matrices

    # now evaluate bayes mix
    with BayesMix(confusion_matrices=confusion_matrices, **net_config) as net:
        import_weights_into_network(net, starting_weights)

        # never evaluate against train data
        evaluation_data['use_trainset'] = False
        measurements, confusion_matrix = evaluate(net, evaluation_data)
        _run.info['measurements'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix

    print('Evaluated Bayes Fusion on {} data:'.format(evaluation_data['dataset']))
    print('total accuracy {:.3f} IoU {:.3f}'.format(measurements['total_accuracy'],
                                                    measurements['mean_IoU']))

    # There seems to be a problem with capturing the print output, flush to be sure
    stdout.flush()
