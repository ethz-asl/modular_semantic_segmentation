from .simple_fcn import SimpleFCN
from .fcn import FCN
from .split_fcn import SplitFCN
from .mix_fcn import MixFCN


def get_model(name):
    if name == 'fcn':
        return FCN
    elif name == 'simple_fcn':
        return SimpleFCN
    elif name == 'split_fcn':
        return SplitFCN
    elif name == 'mix_fcn':
        return MixFCN
    else:
        raise UserWarning('ERROR: Model {} not found'.format(name))
