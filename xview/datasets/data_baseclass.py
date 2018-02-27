import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split

from .augmentation import crop_multiple
from .wrapper import DataWrapper


class DataBaseclass(DataWrapper):
    """A basic, abstract class for splitting data into batches, compliant with DataWrapper
    interface."""

    def __init__(self, trainset, measureset, testset, batchsize, modalities, labelinfo,
                 info=False, single_test_batches=False):
        self.testset, self.validation_set = train_test_split(
            testset, test_size=15, random_state=317243896)
        self.measureset = measureset
        self.trainset = trainset
        self.batch_idx = 0
        self.batchsize = batchsize
        self.modalities = modalities
        self.labelinfo = labelinfo
        self.print_info = info
        # in case images have different sizes, we want single images per test batch
        self.single_test_batches = single_test_batches

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

    def get_set_data(self, setlist, batch_size=None, training_format=False):
        if batch_size is None:
            batch_size = self.batchsize
        size = len(setlist)
        for start_idx in range(0, size, batch_size):
            yield self._get_batch((setlist[idx] for idx
                                   in range(start_idx,
                                            min(start_idx + batch_size, size))),
                                  training_format=training_format)

    def get_train_data(self, batch_size=None, training_format=True):
        """Return generator for train-data."""
        return self.get_set_data(self.trainset, batch_size=batch_size,
                                 training_format=training_format)

    def get_test_data(self, batch_size=None):
        """Return generator for test-data."""
        if self.single_test_batches:
            batch_size = 1
        return self.get_set_data(self.testset, batch_size=batch_size,
                                 training_format=False)

    def get_measure_data(self, batch_size=None):
        """Return generator for test-data."""
        if self.single_test_batches:
            batch_size = 1
        return self.get_set_data(self.measureset, batch_size=batch_size,
                                 training_format=False)

    def get_validation_data(self, num_items=None, batch_size=None):
        """Return a function without arguments that returns a generator for the
        validation data."""
        if num_items is None:
            num_items = len(self.validation_set)
        if self.single_test_batches:
            batch_size = 1
        def data_generator():
            return self.get_set_data(self.validation_set[:num_items],
                                     batch_size=batch_size, training_format=False)
        return data_generator

    def _get_batch(self, items, training_format=True):
        # Dependent on the batchsize, we collect a list of datablobs and group them by
        # modality
        batch = {mod: [] for mod in self.modalities}
        for item in items:
            if self.print_info:
                print(item)
            data = self._get_data(training_format=training_format, **item)
            for mod in self.modalities:
                batch[mod].append(crop_multiple(data[mod]))
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
