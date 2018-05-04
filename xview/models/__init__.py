from .simple_fcn import SimpleFCN
from .bayes_mix import BayesMix
from .dirichlet_mix import DirichletMix
from .progressive_fcn import ProgressiveFCN
from .uncertainty_dirichlet_mix import UncertaintyMix
from .average_mix import AverageFusion
from .variance_mix import VarianceMix
from .adapnet import Adapnet
from .fusion_fcn import FusionFCN
from .bayesian_fcn import BayesianFCN


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
    elif name == 'average_fusion':
        return AverageFusion
    elif name == 'variance_mix':
        return VarianceMix
    elif name == 'progressive_fcn':
        return ProgressiveFCN
    elif name == 'adapnet':
        return Adapnet
    elif name == 'bayesian_fcn':
        return BayesianFCN
    else:
        raise UserWarning('ERROR: Model {} not found'.format(name))
