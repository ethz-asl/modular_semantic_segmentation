import numpy as np

from .data_baseclass import DataBaseclass


class ToyData(DataBaseclass):

    _num_default_classes = 4
    _data_shape_description = {'toy': [2], 'labels': []}

    def __init__(self, **config):
        default_config = {
            'augmentation': {
                'label_flip': False,
                'label_merge': False,
            }
        }
        default_config.update(config)
        self.config = default_config

        labelinfo = {
            0: {'name': 'A', 'color': [255, 0, 0]},
            1: {'name': 'B', 'color': [0, 255, 0]},
            2: {'name': 'C', 'color': [0, 0, 255]},
            3: {'name': 'D', 'color': [128, 128, 0]},
            4: {'name': 'amb', 'color': [0, 0, 0]}
        }

        DataBaseclass.__init__(self,
                               [{'set': 'train'} for _ in range(2000)],
                               [{'set': 'measure'} for _ in range(100)],
                               [{'set': 'test'} for _ in range(1000)],
                               labelinfo,
                               validation_set=[{'set': 'validation'}
                                               for _ in range(1000)])

    def _get_data(self, set, training_format=False):
        blob = {}
        # sample a point
        blob['toy'] = [3 * (np.random.rand() - 0.5) for _ in range(2)]
        # now get the label
        if blob['toy'][0] > 0:
            if blob['toy'][1] > 0:
                blob['labels'] = 0
            else:
                blob['labels'] = 1
        else:
            if blob['toy'][1] > 0:
                blob['labels'] = 2
            else:
                blob['labels'] = 3

        if training_format:
            if self.config['augmentation'].get('label_flip', False):
                c1, c2, p = self.config['augmentation']['label_flip']
                if p < np.random.rand():
                    if blob['labels'] == c1:
                        blob['labels'] = c2
                    elif blob['labels'] == c2:
                        blob['labels'] = c1
            if self.config['augmentation'].get('label_merge', False):
                c1, c2 = self.config['augmentation']['label_merge']
                if blob['labels'] == c2:
                    blob['labels'] = c1
        return blob
