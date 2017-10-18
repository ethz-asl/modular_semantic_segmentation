from .synthia import Synthia


def get_dataset(name, config):
    if name == 'synthia':
        return Synthia(**config)
    else:
        raise UserWarning('ERROR: Dataset {} not found'.format(name))
