from .synthia import Synthia
from .raw_synthia import Synthia as SynthiaRaw

def get_dataset(name, config):
    if name == 'synthia':
        return Synthia(**config)
    elif name == 'raw_synthia':
        return SynthiaRaw(**config)
    else:
        raise UserWarning('ERROR: Dataset {} not found'.format(name))
