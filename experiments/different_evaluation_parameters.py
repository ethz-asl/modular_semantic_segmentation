from sacred import Experiment
from experiments.utils import get_mongo_observer
from experiments.evaluation import evaluate, import_weights_into_network
from xview.models import get_model
from copy import deepcopy


ex = Experiment()
ex.observers.append(get_mongo_observer())


def parameter_combinations(search_parameters, net_config):
    # We want to tets all the different combinations of the search_parameters, therefore,
    # we create a list of the different network configurations.
    configs_to_test = [net_config]
    for parameter, values in search_parameters.items():
        # For all existing configs_to_test, we create one copy for every tested value of
        # this parameter.
        new_configs_to_test = []
        for config in configs_to_test:
            for value in values:
                new_config = deepcopy(config)
                new_config[parameter] = value
                new_configs_to_test.append(new_config)
        configs_to_test = new_configs_to_test
    return configs_to_test


@ex.automain
def main(starting_weights, modelname, net_config, evaluation_data, search_parameters,
         _run):
    model = get_model(modelname)

    # Get different configs we will test
    configs_to_test = parameter_combinations(search_parameters, net_config)

    results = []
    for test_parameters in configs_to_test:
        with model(**test_parameters) as net:
            import_weights_into_network(net, starting_weights)
            measurements, _ = evaluate(net, evaluation_data)

        # put results and parameters all in one dict
        result = {}
        result.update(test_parameters)
        result.update(measurements)
        results.append(result)

    # Results is not a list of dictionaries where all keys match. For convenience (e.g.
    # to import the measurements into a pandas DataFrame, we convert it into a dict of
    # lists), see:
    # [https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists]
    _run.info['results'] = dict(zip(results[0], zip(*[r.values() for r in results])))
