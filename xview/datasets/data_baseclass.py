import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle

from .wrapper import DataWrapper


class DataBaseclass(DataWrapper):
    """A basic, abstract class for splitting data into batches, compliant with DataWrapper
    interface."""

    def __init__(self, trainset, testset, batchsize, modalities, labelinfo):
        self.testset, self.validation_set = train_test_split(testset, test_size=10)
        self.trainset = trainset
        self.batch_idx = 0
        self.batchsize = batchsize
        self.modalities = modalities
        self.labelinfo = labelinfo

        shuffle(self.trainset)

    def _get_data(self, **kwargs):
        """Returns data for one item in trainset or testset. kwargs is the unfolded dict
        from the trainset or testset list
        # (it is called as self._get_data(one_hot=something, **testset[some_idx]))
        """
        raise NotImplementedError

    def _next_batch_idx(self):
        """Increments the index of the next training item. Makes sure the index is reset
        as all training items were used once."""
        self.batch_idx = self.batch_idx + 1
        if self.batch_idx == len(self.trainset):
            self.batch_idx = 0
        return self.batch_idx

    def next(self):
        """As specified by DataWrapper, returns a new training batch."""
        return self._get_batch([self.trainset[self._next_batch_idx()]
                                for _ in range(self.batchsize)])

    def get_train_data(self, batch_size=None):
        """Return generator for train-data."""
        if batch_size is None:
            batch_size = self.batchsize
        trainset_size = len(self.trainset)
        for start_idx in range(0, trainset_size, batch_size):
            yield self._get_batch((self.trainset[idx] for idx in range(start_idx,
                                   min(start_idx + batch_size, trainset_size))),
                                  training_format=False)

    def get_test_data(self, batch_size=None):
        """Return generator for test-data."""
        if batch_size is None:
            batch_size = self.batchsize
        testset_size = len(self.testset)
        for start_idx in range(0, testset_size, batch_size):
            yield self._get_batch((self.testset[idx] for idx in range(start_idx,
                                   min(start_idx + batch_size, testset_size))),
                                  training_format=False)

    def get_validation_data(self, num_items=None, batch_size=None):
        """Return a function without arguments that returns a generator for the
        validation data."""
        if num_items is None:
            num_items = len(self.validation_set)
        if batch_size is None:
            batch_size = self.batchsize

        def data_generator():
            for start_idx in range(0, num_items, batch_size):
                yield self._get_batch((self.testset[idx] for idx in range(start_idx,
                                       min(start_idx + batch_size, num_items))),
                                      training_format=False)
        return data_generator

    def _get_batch(self, items, training_format=True):
        # Dependent on the batchsize, we collect a list of datablobs and group them by
        # modality
        batch = {mod: [] for mod in self.modalities}
        for item in items:
            data = self._get_data(training_format=training_format, **item)
            for mod in self.modalities:
                batch[mod].append(data[mod])
        # Now translate lists of arrays into arrays with first dimension the batch index
        # for each modality.
        for mod in self.modalities:
            batch[mod] = np.stack(batch[mod])
        return batch

    def coloured_labels(self, labels):
        """Return a coloured picture according to set label colours."""
        # To efficiently map class label to color, we create a lookup table
        lookup = np.array([self.labelinfo[i]['color']
                           for i in range(max(self.labelinfo.keys()) + 1)]).astype(int)
        return np.array(lookup[labels[:, :]]).astype('uint8')
