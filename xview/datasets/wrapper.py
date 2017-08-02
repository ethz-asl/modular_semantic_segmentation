from abc import ABCMeta, abstractmethod


class DataWrapper:
    """Interface for access to datasets."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def next(self):
        """Returns next minibatch for training."""
        return NotImplementedError
