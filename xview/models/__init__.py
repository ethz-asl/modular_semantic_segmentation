from .simple_fcn import SimpleFCN
from .mix_fcn import MixFCN
from .progressive_fcn import ProgressiveFCN
from .adapnet import Adapnet


def get_model(name):
    if name == 'simple_fcn':
        return SimpleFCN
    elif name == 'mix_fcn':
        return MixFCN
    elif name == 'progressive_fcn':
        return ProgressiveFCN
    elif name == 'adapnet':
        return Adapnet
    else:
        raise UserWarning('ERROR: Model {} not found'.format(name))
