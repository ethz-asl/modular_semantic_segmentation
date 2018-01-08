import numpy as np

from .wrapper import DataWrapper
from .synthia_rand import SynthiaRand
from .cityscapes import Cityscapes


datasets_dict = {'synthiarand': SynthiaRand, 'cityscapes': Cityscapes}


class MixedData(DataWrapper):

    def __init__(self, **data_config):
        config = {
            'datasets_train': None,
            'dataset_eval': None,
            'batch_distr': None,
            'preprocessing': {'type': 'offline'}
        }
        config.update(data_config)

        if not config['datasets_train']:
            raise UserWarning('Need to specify training datasets.')
        if not config['dataset_eval']:
            raise UserWarning('Need to specify one evaluation dataset.')
        if config['batch_distr'] \
           and len(config['batch_distr']) != len(config['datasets_train']):
            raise UserWarning('Batch distribution must specify all training datasets.')

        if not config['batch_distr']:
            config['batch_distr'] = [1]*len(config['datasets_train'])

        datasets = {}
        for d, w in zip(config['datasets_train'], config['batch_distr']):
            datasets[d] = datasets_dict[d](
                    batchsize=w,
                    no_test=True if d != config['dataset_eval'] else False,
                    **config)
        if config['dataset_eval'] not in datasets:
            datasets[config['dataset_eval']] = \
                    datasets_dict[config['dataset_eval']](**config)

        self.config = config
        self.datasets = datasets
        self.modalities = datasets.values()[0].modalities

    def next(self):
        """As specified by DataWrapper, returns a new training batch."""
        data = [self.datasets[d].next() for d in self.config['datasets_train']]
        return {mod: np.concatenate([d[mod] for d in data]) for mod in self.modalities}

    def get_test_data(self, batch_size=10):
        """Return generator for test-data."""
        return self.datasets[self.config['dataset_eval']].get_test_data(batch_size)

    def get_validation_data(self, num_items=None):
        """Return the test-data in one big batch."""
        return self.datasets[self.config['dataset_eval']].get_test_data(num_items)

    def coloured_labels(self, labels):
        """Return a coloured picture according to set label colours."""
        return self.datasets[self.config['dataset_eval']].coloured_labels(labels)

