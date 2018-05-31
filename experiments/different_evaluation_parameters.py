from sacred import Experiment
from experiments.utils import get_mongo_observer
from experiments.evaluation import evaluate, import_weights_into_network
from xview.models import get_model
from copy import deepcopy
import os
from tqdm import tqdm


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


def grid_search(evaluation, search_parameters, net_config):
    """Performs grid-search on the search-parameters.

    Args:
        evaluation: takes a set of parameters as input and returns a nested dict
            of measurements for this grid point
        search_parameters: dict of lists for parameters and their values that will be
            tested in all combinations
        net_config:
            standard parameters that are added to the search parameters
    Returns:
        a dict with keys from search_parameters and the evaluation results,
        where values of parameters and evaluation are sorted in lists for the
        different gridpoints
    """
    # get different configurations that will be tested
    configs_to_test = parameter_combinations(search_parameters, net_config)

    # collect results in a list
    results = {}
    for test_parameters in tqdm(configs_to_test, ascii=True):
        # result is a dict of parameters and measurements
        for key in test_parameters:
            results.setdefault(key, []).append(test_parameters[key])
        result = evaluation(test_parameters)

        # traverse the nested dict of result and append the values to results
        def append_deep_value(add_to, val):
            for key, inner_val in val.items():
                if isinstance(inner_val, dict):
                    append_deep_value(add_to.setdefault(key, {}), inner_val)
                else:
                    add_to.setdefault(key, []).append(inner_val)
        append_deep_value(results, result)
    return results


ex = Experiment()
ex.observers.append(get_mongo_observer())


@ex.automain
def main(starting_weights, modelname, net_config, evaluation_data, search_parameters,
         _run):
    model = get_model(modelname)

    def evaluation(parameters):
        with model(**parameters) as net:
            import_weights_into_network(net, starting_weights)
            measurements, _ = evaluate(net, evaluation_data)
        return measurements

    _run.info['results'] = grid_search(evaluation, search_parameters, net_config)
