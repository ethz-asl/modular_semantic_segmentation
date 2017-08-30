import numpy as np
from sklearn.utils import shuffle

from .wrapper import DataWrapper


class DataBaseclass(DataWrapper):
    """A basic, abstract class for splitting data into batches, compliant with DataWrapper
    interface."""

    def __init__(self, trainset, testset, batchsize, modalities):
        self.testset = testset
        self.trainset = shuffle(trainset)
        self.batch_idx = 0
        self.batchsize = batchsize
        self.modalities = modalities

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
        # Dependent on the batchsize, we collect a list of datablobs and group them by
        # modality
        batch = {mod: [] for mod in self.modalities}
        i = 1
        while i <= self.batchsize:
            item = self._get_data(**self.trainset[self.batch_idx])
            for mod in self.modalities:
                batch[mod].append(item[mod])
            i = i + 1
            self._increment_batch_idx()
        # Now translate lists of arrays into arrays with first dimension the batch index
        # for each modality.
        for mod in self.modalities:
            batch[mod] = np.stack(batch[mod])
        return batch

    def get_test_data(self):
        """Return the test-data in one big batch."""
        testdata = {mod: [] for mod in self.modalities}
        for item in self.testset:
            data = self._get_data(**item)
            for mod in self.modalities:
                testdata[mod].append(data[mod])
        # Now translate lists of arrays into arrays with first dimension the batch index
        # for each modality.
        for mod in self.modalities:
            testdata[mod] = np.stack(testdata[mod])
        return testdata
