from .synthia import Synthia


def get_dataset(name):
    if name == 'synthia':
        return Synthia
    else:
        raise UserWarning('ERROR: Dataset {} not found'.format(name))
