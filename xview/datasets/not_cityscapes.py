from xview.settings import DATA_BASEPATH
from os import path, environ
from xview.datasets import Cityscapes
import cv2
import numpy as np
from tqdm import tqdm
import tarfile
from .augmentation import augmentate
from .data_baseclass import DataBaseclass


def get_dataset(name):
    """have to reimplement as there is an import loop when using __init__"""
    if name == 'cityscapes':
        return Cityscapes


class AddRandomObjects(DataBaseclass):

    _data_shape_description = {'rgb': (None, None, 3), 'labels': (None, None)}
    _num_default_classes = 2

    def __init__(self, add_to_dataset='cityscapes', halfsize=True, augmentation=False,
                 in_memory=False, **config):
        self.base_path = path.join(DATA_BASEPATH, 'amsterdam_object_lib')
        if not path.exists(self.base_path):
            message = 'ERROR: Path to CITYSCAPES dataset does not exist.'
            print(message)
            raise IOError(1, message, self.base_path)

        self.config = {
            'halfsize': halfsize,
            'augmentation': augmentation,
            'in_memory': in_memory
        }

        print('INFO: Loading Base Dataset')
        self.base_dataset = get_dataset(add_to_dataset)(in_memory=in_memory, **config)

        if in_memory:
            print('INFO loading dataset into memory')
            # first load the tarfile into a closer memory location, then load all the
            # images
            tar = tarfile.open(path.join(self.base_path, 'amsterdam_lib.tar.gz'))
            localtmp = environ['TMPDIR']
            tar.extractall(path=localtmp)
            tar.close()
            self.base_path = localtmp
            self.objects = {num: self._load_object(num)
                            for num in tqdm(range(251, 1001), ascii=True)}

        DataBaseclass.__init__(self, self.base_dataset.trainset,
                               self.base_dataset.measureset,
                               self.base_dataset.testset,
                               {0: {'name': 'in-distribution'},
                                1: {'name': 'out-of-distribution'}},
                               validation_set=self.base_dataset.validation_set,
                               num_classes=self.base_dataset._num_default_classes)

    def _load_object(self, object_name):
        obj = cv2.imread(path.join(self.base_path, '{0}/{0}_c.png'.format(object_name)))

        if self.config['halfsize']:
            h, w, _ = obj.shape
            obj = cv2.resize(obj, (h // 2, w // 2))
        return obj

    def _get_data(self, training_format=False, **kwargs):
        # load image from base dataset
        img = self.base_dataset._get_data(training_format=False, **kwargs)['rgb']

        # get a random object
        num = np.random.randint(251, 1000)
        if self.config['in_memory']:
            obj = self.objects[num].copy()
        else:
            obj = self._load_object(num)
        h, w, _ = obj.shape

        # sample a random location where to put the object in the image
        img_h, img_w, _ = img.shape
        top = np.random.randint(img_h - h)
        left = np.random.randint(img_w - w)
        # create an overlay image with the object of the same size as the underlying
        # image
        obj = cv2.copyMakeBorder(obj, top, img_h - top - h, left, img_w - left - w,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # to filter out the object from the dark background, we consider everything
        # darker than (30, 30, 30) as background
        # this is not a perfect filter, but it works reasonably well and we want to
        # create out-of-distribution data anyways
        blob = {
            'rgb': np.where(np.all(obj < 30, axis=2, keepdims=True), img, obj),
            'labels': (1 - np.all(obj < 30, axis=2))
        }

        if training_format:
            blob = augmentate(blob, **self.config.augmentation)
        return blob
