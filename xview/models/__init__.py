from .simple_fcn import SimpleFCN
from .bayes_mix import BayesMix
from .dirichlet_mix import DirichletMix
from .progressive_fcn import ProgressiveFCN
from .uncertainty_dirichlet_mix import UncertaintyMix
from .average_mix import AverageMix
from .adapnet import Adapnet
from .fusion_fcn import FusionFCN


def get_model(name):
    if name == 'fcn':
        return SimpleFCN
    elif name == 'fusion_fcn':
        return FusionFCN
    elif name == 'bayes_mix':
        return BayesMix
    elif name == 'dirichlet_mix':
        return DirichletMix
    elif name == 'uncertainty_mix':
        return UncertaintyMix
    elif name == 'average_mix':
        return AverageMix
    elif name == 'progressive_fcn':
        return ProgressiveFCN
    elif name == 'adapnet':
        return Adapnet
    else:
        raise UserWarning('ERROR: Model {} not found'.format(name))
