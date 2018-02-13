import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
import random
import cv2
from imgaug import augmenters as ia

from .wrapper import DataWrapper


def crop_multiple(data, multiple_of=16):
    """Force the array dimension to be multiple of the given factor.

    Args:
        data: a >=2-dim array, first 2 dims will be cropped
        multiple_of: the factor, as an int
    Returns:
        cropped array
    """
    h, w = data.shape[0], data.shape[1]
    h_c, w_c = [d - (d % multiple_of) for d in [h, w]]
    if h_c != h or w_c != w:
        return data[:h_c, :w_c, ...]
    else:
        return data


def augmentate(blob, scale=False, crop=False, hflip=False, vflip=False, gamma=False,
               contrast=False, brightness=False, rotate=False, shear=False):
    """Perform data-augmentations on all modalities of an image blob.

    Args:
        if set, every argument is prepented with a individual probability that determines
            whether or not the given augmentation is performed
        scale (list of 2 values or False): Scale the image by a factor from the given
            interval
        crop (int or False): crop the image (after scaling) to a square of the given size
        vflip (bool): vertically flip the image
        hflip (bool): horizontally flip the image
        gamma (list of 2 values or False): apply gamma correction/noise with a random
            factor from the given interval
    Returns:
        augemted image blob
    """
    modalities = list(blob.keys())

    # find out whether or not we are doing cropping later on
    do_crop = False
    if crop and crop[0] > random.random():
        do_crop = True

    if scale and do_crop and scale[0] > random.random():
        h, w, _ = blob[modalities[0]].shape
        min_scale = crop[1] / float(min(h, w))
        k = random.uniform(max(min_scale, scale[1]), scale[2])

        # RGB is resized using bilinear interpolation, all other modalities should use
        # nearest neighbour as their values do not necessarily behave like rgb
        if 'rgb' in blob:
            blob['rgb'] = cv2.resize(blob['rgb'], None, fx=k, fy=k)
        for m in (m for m in modalities if m != 'rgb'):
            blob[m] = cv2.resize(blob[m], None, fx=k, fy=k,
                                 interpolation=cv2.INTER_NEAREST)

    if rotate and do_crop and rotate[0] > random.random():
        deg = np.random.randint(rotate[1], rotate[2])
        rotator = ia.Affine(rotate=deg)
        for m in modalities:
            blob[m] = rotator.augment_images(blob[m])

    if shear and do_crop and shear[0] > random.random():
        h, w, _ = blob[modalities[0]].shape
        shear_px = np.random.randint(shear[1] * w, shear[2] * w) \
            * np.random.choice([-1, 1])
        augmentation = ia.Affine(shear=shear_px)
        for m in modalities:
            blob[m] = augmentation.augment_images(blob[m])

    if do_crop:
        h, w, _ = blob[modalities[0]].shape
        h_c = random.randint(0, h - crop[1])
        w_c = random.randint(0, w - crop[1])
        for m in modalities:
            blob[m] = blob[m][h_c:h_c+crop[1], w_c:w_c+crop[1], ...]

    if hflip and hflip > random.random() and np.random.choice([0, 1]):
        for m in modalities:
            blob[m] = np.flip(blob[m], axis=0)

    if vflip and vflip > random.random() and np.random.choice([0, 1]):
        for m in modalities:
            blob[m] = np.flip(blob[m], axis=1)

    if contrast and 'rgb' in modalities:
        augmentation = ia.Sometimes(contrast[0], ia.ContrastNormalization((contrast[1],
                                                                           contrast[2])))
        blob['rgb'] = augmentation.augment_images(blob['rgb'])

    if brightness and 'rgb' in modalities:
        augmentation = ia.Sometimes(brightness[0],
                                    ia.Add((brightness[1], brightness[2])))
        blob['rgb'] = augmentation.augment_images(blob['rgb'])

    if gamma and 'rgb' in modalities and gamma[0] > random.random():
        # gamma noise should only be applied to rgb
        k = random.uniform(gamma[1], gamma[2])
        lut = np.array([((i / 255.0) ** (1/k)) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
        blob['rgb'] = lut[blob['rgb']]

    return blob


class DataBaseclass(DataWrapper):
    """A basic, abstract class for splitting data into batches, compliant with DataWrapper
    interface."""

    def __init__(self, trainset, testset, batchsize, modalities, labelinfo,
                 info=False):
        self.testset, self.validation_set = train_test_split(
            testset, test_size=15, random_state=317243896)
        self.trainset = trainset
        self.batch_idx = 0
        self.batchsize = batchsize
        self.modalities = modalities
        self.labelinfo = labelinfo
        self.print_info = info

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
            yield self._get_batch((self.trainset[idx] for idx
                                   in range(start_idx,
                                            min(start_idx + batch_size, trainset_size))),
                                  training_format=True)

    def get_test_data(self, batch_size=None):
        """Return generator for test-data."""
        if batch_size is None:
            batch_size = self.batchsize
        testset_size = len(self.testset)
        for start_idx in range(0, testset_size, batch_size):
            yield self._get_batch((self.testset[idx] for idx
                                   in range(start_idx,
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
                yield self._get_batch((self.testset[idx] for idx
                                       in range(start_idx,
                                                min(start_idx + batch_size, num_items))),
                                      training_format=False)
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
