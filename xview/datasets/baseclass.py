import numpy as np
from sklearn.utils import shuffle

from .wrapper import DataWrapper


class DataBaseclass(DataWrapper):
    """A basic, abstract class for splitting data into batches, compliant with DataWrapper
    interface."""

    def __init__(self, trainset, testset, batchsize):
        self.testset = testset
        self.trainset = shuffle(trainset)
        self.batch_idx = 0
        self.batchsize = batchsize

    def _get_data(self, **kwargs):
        """Returns data for one item in trainset or testset. kwargs is the unfolded dict
        from the trainset or testset list
        # (it is called as self._get_data(**testset[some_idx]))
        """
        raise NotImplementedError

    def _increment_batch_idx(self):
        """Increments the index of the next training item. Makes sure the index is reset
        as all training items were used once."""
        self.batch_idx = self.batch_idx + 1
        if self.batch_idx == len(self.trainset):
            self.batch_idx = 0

    def next(self):
        """As specified by DataWrapper, returns a new training batch."""
        batch = {}
        i = 0
        while i < self.batchsize:
            item = self._get_data(**self.trainset[self.batch_idx])
            for key in item:
                if key not in batch:
                    # this is the first item, generate lists for every key
                    batch[key] = [item[key]]
                batch[key].append(item[key])
            i = i + 1
            self._increment_batch_idx()
        # Now translate lists of arrays into arrays with first dimension the batch index
        # for each key.
        for key in batch:
            batch[key] = np.array(batch[key])
        return batch

    def get_test_data(self):
        """Return the test-data in one big batch."""
        testdata = {}
        for item in self.testset:
            data = self._get_data(**item)
            for key in data:
                if key not in testdata:
                    # This is the first item and we have to add the key to the testdata
                    # dict.
                    testdata[key] = item[key]
                else:
                    testdata[key].append(item[key])
        return testdata
