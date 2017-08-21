from sklearn.utils import shuffle

from .wrapper import DataWrapper


class DataBaseclass(DataWrapper):
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
        if self.batch_idx == len(self.trainset):
            self.batch_idx = 0
        else:
            self.batch_idx = self.batch_idx + 1

    def next(self):
        batch = self._get_data(**self.trainset[self.batch_idx])
        i = 1
        self._increment_batch_idx()
        while i < self.batchsize:
            item = self._get_data(**self.trainset[self.batch_idx])
            for key in batch:
                batch[key].append(item[key])
            i = i + 1
            self._increment_batch_idx()
        return batch

    def get_test_data(self):
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
