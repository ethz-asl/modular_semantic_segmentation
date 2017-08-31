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

    def get_test_data(self, num_items=None):
        """Return the test-data in one big batch."""
        if num_items is None:
            return self._get_batch(self.testset, one_hot=False)
        else:
            # generate given number of random indizes
            idx = np.random.choice(range(len(self.testset)), size=num_items,
                                   replace=False)
            return self._get_batch(self.testset[i] for i in idx, one_hot=False)

    def _get_batch(self, items, one_hot=True):
        # Dependent on the batchsize, we collect a list of datablobs and group them by
        # modality
        batch = {mod: [] for mod in self.modalities}
        for item in items:
            data = self._get_data(one_hot=one_hot, **item)
            for mod in self.modalities:
                batch[mod].append(data[mod])
        # Now translate lists of arrays into arrays with first dimension the batch index
        # for each modality.
        for mod in self.modalities:
            batch[mod] = np.stack(batch[mod])
        return batch
