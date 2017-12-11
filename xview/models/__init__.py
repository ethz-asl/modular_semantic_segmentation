from .simple_fcn import SimpleFCN
from .bayes_mix import BayesMix
from .dirichlet_mix import DirichletMix
from .progressive_fcn import ProgressiveFCN
from .adapnet import Adapnet


def get_model(name):
    if name == 'simple_fcn':
        return SimpleFCN
    elif name == 'bayes_mix':
        return BayesMix
    elif name == 'dirichlet_mix':
        return DirichletMix
    elif name == 'progressive_fcn':
        return ProgressiveFCN
    elif name == 'adapnet':
        return Adapnet
    else:
        raise UserWarning('ERROR: Model {} not found'.format(name))
