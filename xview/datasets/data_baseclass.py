import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

from .augmentation import crop_multiple
from .wrapper import DataWrapper


class DataBaseclass(DataWrapper):
    """A basic, abstract class for splitting data into batches, compliant with DataWrapper
    interface."""

    def __init__(self, trainset, measureset, testset, num_classes, data_description,
                 labelinfo, info=False, single_test_batches=False):
        self.trainset, self.validation_set = train_test_split(
            trainset, test_size=15, random_state=317243896)
        self.measureset = measureset
        self.testset = testset
        self.num_classes = num_classes
        self.data_description = data_description
        self.modalities = list(data_description.keys())
        self.labelinfo = labelinfo
        self.print_info = info
        shuffle(self.trainset)

    def _get_data(self, **kwargs):
        """Returns data for one item in trainset or testset. kwargs is the unfolded dict
        from the trainset or testset list
        # (it is called as self._get_data(one_hot=something, **testset[some_idx]))
        """
        raise NotImplementedError

    def _get_batch(self, items, training_format=False):
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

    def _get_tf_dataset(self, setlist):
        def data_generator():
            for item in setlist:
                data = self._get_data(training_format=False, **item)
                for m in self.modalities:
                    data[m] = crop_multiple(data[m])
                yield data

        return tf.data.Dataset.from_generator(data_generator,
                                              *self.get_data_description()[:2])

    def get_trainset(self, tf_dataset=True, training_format=False):
        """Return trainingset. By default as tf.data.dataset, otherwise as numpy array.
        """
        if not tf_dataset:
            return self._get_batch(self.trainset, training_format=training_format)
        return self._get_tf_dataset(self.trainset)

    def get_testset(self, tf_dataset=True):
        """Return testset. By default as tf.data.dataset, otherwise as numpy array."""
        if not tf_dataset:
            return self._get_batch(self.testset)
        return self._get_tf_dataset(self.testset)

    def get_measureset(self, tf_dataset=True):
        """Return measureset. By default as tf.data.dataset, otherwise as numpy array."""
        if not tf_dataset:
            return self._get_batch(self.measureset)
        return self._get_tf_dataset(self.measureset)

    def get_validation_set(self, num_items=None, tf_dataset=True):
        """Return testset. By default as tf.data.dataset, otherwise as numpy array."""
        if num_items is None:
            num_items = len(self.validation_set)
        if not tf_dataset:
            return self._get_batch(self.validation_set[:num_items])
        return self._get_tf_dataset(self.validation_set[:num_items])

    def coloured_labels(self, labels):
        """Return a coloured picture according to set label colours."""
        # To efficiently map class label to color, we create a lookup table
        lookup = np.array([self.labelinfo[i]['color']
                           for i in range(max(self.labelinfo.keys()) + 1)]).astype(int)
        return np.array(lookup[labels[:, :]]).astype('uint8')

    def get_data_description(self):
        return ({'labels': tf.int32, **{m: tf.float32 for m in self.modalities
                                        if not m == 'labels'}},
                self.data_description,
                self.num_classes)
