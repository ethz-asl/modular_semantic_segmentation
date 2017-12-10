from .simple_fcn import SimpleFCN
from .fcn import FCN
from .split_fcn import SplitFCN
from .simple_mix_fcn import MixFCN as BayesMix
from .progressive_fcn import ProgressiveFCN


def get_model(name):
    if name == 'fcn':
        return FCN
    elif name == 'simple_fcn':
        return SimpleFCN
    elif name == 'split_fcn':
        return SplitFCN
    elif name == 'bayes_mix':
        return BayesMix
    elif name == 'progressive_fcn':
        return ProgressiveFCN
    else:
        raise UserWarning('ERROR: Model {} not found'.format(name))
