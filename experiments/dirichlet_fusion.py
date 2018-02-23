from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.utils import get_mongo_observer
from experiments.evaluation import import_weights_into_network
from experiments.different_evaluation_parameters import parameter_combinations
from experiments.bayes_fusion import split_test_data
from sklearn.model_selection import train_test_split
from xview.datasets import get_dataset
from xview.models import DirichletMix
from sys import stdout


ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_mongo_observer())


@ex.command
def test_parameters(net_config, evaluation_data, starting_weights, search_parameters,
                    _run):
    # get the different configs we will test
    configs_to_test = parameter_combinations(search_parameters, net_config)

    # load data
    data, measure_data, _ = split_test_data(evaluation_data)
    search_data, search_validation = train_test_split(measure_data, test_size=.5,
                                                      random_state=1)
    # get sufficient statistic
    with DirichletMix(**configs_to_test[0]) as net:
        import_weights_into_network(net, starting_weights)
        sufficient_statistic = net._get_sufficient_statistic(
            data.get_set_data(search_data))

    # Not test all the parameters
    results = []
    for test_parameters in configs_to_test:
        with DirichletMix(**test_parameters) as net:
            net._fit_sufficient_statistic(*sufficient_statistic)
            import_weights_into_network(net, starting_weights)
            measurements, _ = net.score(data.get_set_data(search_validation))

            # put results and parameters all in one dict
            result = {}
            result.update(test_parameters)
            result.update(measurements)
            results.append(result)

    # Results is not a list of dictionaries where all keys match. For convenience (e.g.
    # to impor the measurements into a pandas DataFrame, we convert it into a dict of
    # lists), see:
    # [https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists]
    _run.info['results'] = dict(zip(results[0], zip(*[r.values() for r in results])))


@ex.automain
def fit_and_evaluate(net_config, evaluation_data, starting_weights, _run):
    """Load weigths from trainign experiments and evalaute network against specified
    data."""
    data, measure_data, test_data = split_test_data(evaluation_data)
    _, measure_data = train_test_split(measure_data, test_size=.5, random_state=1)

    with DirichletMix(**net_config) as net:
        import_weights_into_network(net, starting_weights)

        dirichlet_params = net.fit(data.get_set_data(measure_data))

        # import weights again has fitting created new graph
        import_weights_into_network(net, starting_weights)

        measurements, confusion_matrix = net.score(data.get_set_data(test_data))
        _run.info['measurements'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix
        _run.info['dirichlet_params'] = dirichlet_params

    print('Evaluated Dirichlet Fusion on {} data:'.format(evaluation_data['dataset']))
    print('total accuracy {:.3f} IoU {:.3f}'.format(measurements['total_accuracy'],
                                                    measurements['mean_IoU']))
    # There seems to be a problem with capturing the print output, flush to be sure
    stdout.flush()
