from sacred import Experiment
from experiments.utils import ExperimentData, get_mongo_observer
from experiments.evaluation import evaluate, import_weights_into_network
from experiments.different_evaluation_parameters import parameter_combinations
from xview.datasets import get_dataset
from xview.models.dirichlet_mix import DirichletMix
from sys import stdout
import os


ex = Experiment()
ex.observers.append(get_mongo_observer())

@ex.command
def test_parameters(net_config, evaluation_data, starting_weights, search_paramters,
                    _run):
    # get the different configs we will test
    configs_to_test = parameter_combinations(search_paramters, net_config)

    # generate sufficient statistic
    some_config = configs_to_test[0]

    dataset_params = {key: val for key, val in evaluation_data.items()
                      if key not in ['dataset', 'use_trainset']}
    dataset_params['batchsize'] = 1
    # Load the dataset, we expect config to include the arguments
    data = get_dataset(evaluation_data['dataset'], dataset_params)
    batches = data.get_train_data(batch_size=6)
    with MixFCN(**some_config) as net:
        import_weights_into_network(net, starting_weights)
        sufficient_statistic = net._get_sufficient_statistic(batches)

    data = load_data(evaluation_data)
    validation_data = data.get_validation_data()

    # Not test all the parameters
    results = []
    for test_parameters in configs_to_test:
        with MixFCN(**test_parameters) as net:
            net._get_sufficient_statistic(*sufficient_statistic)
            import_weights_into_network(net, starting_weights)
            measurements, _ = net.fit(validation_data())

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
    output_dir = '/tmp/mix_fcn'

    with DirichletMix(output_dir=output_dir, **net_config) as net:
        import_weights_into_network(net, starting_weights)

        # Measure the single experts against the trainingset.
        dataset_params = {key: val for key, val in evaluation_data.items()
                          if key not in ['dataset', 'use_trainset']}
        dataset_params['batchsize'] = 1
        # Load the dataset, we expect config to include the arguments
        data = get_dataset(evaluation_data['dataset'], dataset_params)
        if evaluation_data['use_trainset']:
            net.fit(data.get_train_data())
        else:
        #net.fit(data.get_validation_data(num_items=1))
            net.fit(data.get_validation_data()())


        # import weights again has fitting created new graph
        import_weights_into_network(net, starting_weights)

        measurements, confusion_matrix = evaluate(net, evaluation_data)
        _run.info['measurements'] = measurements
        _run.info['confusion_matrix'] = confusion_matrix

        # To end the experiment, we collect all produced output files and store them.
        for filename in os.listdir(output_dir):
            ex.add_artifact(os.path.join(output_dir, filename))
