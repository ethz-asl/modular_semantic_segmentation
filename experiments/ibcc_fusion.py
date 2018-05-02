from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.utils import get_mongo_observer
from experiments.evaluation import import_weights_into_network
from xview.datasets import get_dataset
from xview.models import get_model
import numpy as np
from os import path, mkdir
from copy import deepcopy


ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_mongo_observer())


@ex.automain
def collect_data(net_config, dataset, starting_weights, save_to, _run):
    data = get_dataset(**dataset)
    model = get_model(net_config['expert_model'])

    predictions = {}
    for expert in net_config['prefixes']:
        model_config = deepcopy(net_config)
        model_config['modality'] = expert
        model_config['prefix'] = net_config['prefixes'][expert]
        with model(data_description=data.get_data_description(),
                   **model_config) as net:
            import_weights_into_network(net, starting_weights[model_config['prefix']])
            predictions['measure_%s' % expert] = net.predict(data.get_measureset())
            predictions['test_%s' % expert] = net.predict(data.get_testset())

    # add also gt labels
    predictions['measure_gt'] = data.get_measureset(tf_dataset=False)['labels']
    predictions['test_gt'] = data.get_testset(tf_dataset=False)['labels']

    # outpath = path.join(save_to, _run._id)
    outpath = save_to
    if not path.exists(outpath):
        mkdir(outpath)
    np.savez_compressed(path.join(outpath, 'predictions.npz'), **predictions)
