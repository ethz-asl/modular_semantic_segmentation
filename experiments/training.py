from sacred import Experiment
from experiments.utils import get_mongo_observer, load_data
from experiments.evaluation import evaluate, import_weights_into_network
from sacred.utils import TimeoutInterrupt
from xview.models import get_model
import os
import shutil


def create_directories(run_id, experiment):
    """
    Make sure directories for storing diagnostics are created and clean.

    Args:
        run_id: ID of the current sacred run, you can get it from _run._id in a captured
            function.
        experiment: The sacred experiment object
    Returns:
        The path to the created output directory you can store your diagnostics to.
    """
    root = '/tmp/sacred/training'
    # create temporary directory for output files
    if not os.path.exists(root):
        os.makedirs(root)
    # The id of this experiment is stored in the magical _run object we get from the
    # decorator.
    output_dir = '{}/{}'.format(root, run_id)
    if os.path.exists(output_dir):
        # Directory may already exist if run_id is None (in case of an unobserved test-run)
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # Tell the experiment that this output dir is also used for tensorflow summaries
    experiment.info.setdefault("tensorflow", {}).setdefault("logdirs", [])\
        .append(output_dir)
    return output_dir


ex = Experiment()
ex.observers.append(get_mongo_observer())


def train_network(net, output_dir, data_config, num_iterations, starting_weights,
                  experiment, additional_eval_data={}):
    """\
    Train a network on a given dataset.

    Args:
        net: An instance of a `base_model` class.
        output_dir: A directory path. This function will add all files foudn at this path
            as artifacts to the experiment.
        data_config: A config-dict for data containing all initializer arguments and the
            dataset-name at key 'dataset'.
        num_iterations: The numbe rof training iterations
        starting_weights: Desriptor for weight sto load into network. If not false or
            empty, will load weights as described in `evaluation.py`.
        experiment: The current sacred experiment.
    """

    # get validation set
    data = load_data(data_config)
    validation_set = data.get_validation_data(num_items=10, batch_size=1)

    # Train the given network
    if starting_weights:
        import_weights_into_network(net, starting_weights)

    try:
        net.fit(data, num_iterations, validation_data=validation_set,
                additional_eval_data=additional_eval_data)
        timeout = False
    except KeyboardInterrupt:
        print('WARNING: Got Keyboard Interrupt, will save weights and close')
        timeout = True

    # Store the weights into the standard output directory
    net.export_weights()

    # To end the experiment, we collect all produced output files and store them.
    for filename in os.listdir(output_dir):
        experiment.add_artifact(os.path.join(output_dir, filename))

    if timeout:
        raise TimeoutInterrupt


@ex.capture
def train_and_evaluate(net, output_dir, data_config, num_iterations,
                       starting_weights, _run):
    """Train and evaluate a given network."""
    train_network(net, output_dir, data_config, num_iterations, starting_weights, ex)
    measurements, _ = evaluate(net, data_config)
    _run.info['measurements'] = measurements


@ex.automain
def main(modelname, net_config, _run):
    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # create the network
    model = get_model(modelname)
    with model(output_dir=output_dir, **net_config) as net:
        train_and_evaluate(net, output_dir)
