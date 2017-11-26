from .synthia import Synthia
from .freiburg_forest import FreiburgForest


def get_dataset(name, config):
    if name == 'synthia':
        return Synthia(**config)
    elif name == 'freiburg_forest':
        return FreiburgForest(**config)
    else:
        raise UserWarning('ERROR: Dataset {} not found'.format(name))
