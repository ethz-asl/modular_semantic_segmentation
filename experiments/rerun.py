from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.utils import get_mongo_observer
import os
import sys

from experiments.utils import ExperimentData
import experiments.bayes_fusion as bayes_fusion
import experiments.timing as timing
import experiments.training as training
import experiments.evaluation as evaluation
import experiments.different_evaluation_parameters as different_evaluation_parameters
import experiments.dirichlet_fusion as dirichlet_fusion


# map modules against their filenames

module_mapper = {
    'bayes_fusion.py': bayes_fusion,
    'timing.py': timing,
    'training.py': training,
    'evaluation.py': evaluation,
    'different_evaluation_parameters.py': different_evaluation_parameters,
    'dirichlet_fusion.py': dirichlet_fusion,
}

ex = Experiment()
ex.capture_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_mongo_observer())


@ex.main
def rerun(experiment_id, _run):
    # load the old experiment
    old_run = ExperimentData(experiment_id).get_record()

    print('Re-Run of experiment "%s"' % old_run['experiment']['name'])

    # load the experiment function
    module = module_mapper[old_run['experiment']['mainfile']]
    command = getattr(module, old_run['command'])
    config = old_run['config']

    # add run to arguments
    if '_run' in command.__code__.co_varnames:
        config['_run'] = _run
    if 'seed' not in command.__code__.co_varnames:
        config.pop('seed', None)
    # now execute the old command
    command(**config)

    sys.stdout.flush()


if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
