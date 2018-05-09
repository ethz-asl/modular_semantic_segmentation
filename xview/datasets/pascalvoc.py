import numpy as np
from os import path, environ
import cv2
import tarfile
from sklearn.model_selection import train_test_split

from xview.settings import DATA_BASEPATH
from .data_baseclass import DataBaseclass
from .augmentation import augmentate
from tqdm import tqdm


PASCALVOC_BASEPATH = path.join(DATA_BASEPATH, 'pascalvoc')


class PascalVOC(DataBaseclass):

    _data_shape_description = {'rgb': (None, None, 3), 'labels': (None, None)}
    _num_default_classes = 21

    def __init__(self, base_path=PASCALVOC_BASEPATH, in_memory=False, **data_config):

        config = {
            'augmentation': {
                'crop': [1, 240],
                'scale': [.4, 1, 1.5],
                'vflip': .3,
                'hflip': False,
                'gamma': [.4, 0.3, 1.2],
                'rotate': False,
                'shear': False,
                'contrast': [.3, 0.5, 1.5],
                'brightness': [.2, -40, 40]
            },
        }
        config.update(data_config)
        self.config = config

        if not path.exists(base_path):
            message = 'ERROR: Path to CITYSCAPES dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.base_path = base_path

        self.labelinfo = {
            0: {'name': 'background', 'color': [0, 0, 0]},
            1: {'name': 'aeroplane', 'color': [128, 0, 0]},
            2: {'name': 'bicycle', 'color': [0, 128, 0]},
            3: {'name': 'bird', 'color': [128, 128, 0]},
            4: {'name': 'boat', 'color': [0, 0, 128]},
            5: {'name': 'bottle', 'color': [128, 0, 128]},
            6: {'name': 'bus', 'color': [0, 128, 128]},
            7: {'name': 'car', 'color': [128, 128, 128]},
            8: {'name': 'cat', 'color': [64, 0, 0]},
            9: {'name': 'chair', 'color': [192, 0, 0]},
            10: {'name': 'cow', 'color': [64, 128, 0]},
            11: {'name': 'diningtable', 'color': [192, 128, 0]},
            12: {'name': 'dog', 'color': [64, 0, 128]},
            13: {'name': 'horse', 'color': [192, 0, 128]},
            14: {'name': 'motorbike', 'color': [64, 128, 128]},
            15: {'name': 'person', 'color': [192, 128, 128]},
            16: {'name': 'pottedplant', 'color': [0, 64, 0]},
            17: {'name': 'sheep', 'color': [128, 64, 0]},
            18: {'name': 'sofa', 'color': [0, 192, 0]},
            19: {'name': 'train', 'color': [128, 192, 0]},
            20: {'name': 'tvmonitor', 'color': [0, 64, 128]},
        }

        # load training and test sets
        def get_filenames(fileset):
            listfile = path.join(self.base_path, 'ImageSets/Segmentation',
                                 '%s.txt' % fileset)
            with open(listfile, 'r') as f:
                filenames = f.readlines()
            # create dictionaries and remove lineending from strings
            return list(map(lambda x: {'image_name': x[:-1]}, filenames))

        if in_memory:
            print('INFO loading dataset into memory')
            # first load the tarfile into a closer memory location, then load all the
            # images
            tar = tarfile.open(path.join(base_path, 'pascalvoc.tar.gz'))
            localtmp = environ['TMPDIR']
            tar.extractall(path=localtmp)
            tar.close()
            self.base_path = localtmp
            trainset = [{'image': self._load_data(i['image_name'], i['image_path'])}
                        for i in tqdm(get_filenames('train'))]
            testset = [{'image': self._load_data(i['image_name'], i['image_path'])}
                       for i in tqdm(get_filenames('val'))]
        else:
            trainset = get_filenames('train')
            testset = get_filenames('val')

        trainset, measureset = train_test_split(trainset, test_size=0.05,
                                                random_state=4)

        # Intitialize Baseclass
        DataBaseclass.__init__(self, trainset, measureset, testset, self.labelinfo)

    def _load_data(self, image_name):
        blob = {}
        blob['rgb'] = cv2.imread(path.join(self.base_path, 'JPEGImages',
                                           '%s.jpg' % image_name))
        blob['labels'] = cv2.imread(path.join(self.base_path, 'SegmentationClass',
                                              '%s.png' % image_name))
        # map label colors on class indizes

        def map_to_classes(img, labelinfo):
            # based on the nice answer in https://stackoverflow.com/questions/38981912/
            # how-to-map-pixels-r-g-b-in-a-collection-of-images-to-a-distinct-pixel-color
            ravelidx2class = np.zeros(255 ** 3)
            # everything that is not defined in labelinfo (e.g. void) will be mapped onto
            # nan
            ravelidx2class[ravelidx2class == 0] = np.nan
            for key, c in labelinfo.items():
                ravelidx2class[np.ravel_multi_index(c['color'], [255, 255, 255])] = key
            H, W = img.shape[0], img.shape[1]
            img2D = img.reshape(-1, 3)
            ID = np.ravel_multi_index(img2D.T, [255, 255, 255])
            idx_img = ID.reshape(H, W)
            return ravelidx2class[idx_img]

        blob['labels'] = map_to_classes(blob['labels'], self.labelinfo)
        return blob

    def _get_data(self, image_name=False, image=False, training_format=False):
        """Returns data for one given image number from the specified sequence."""
        if not image_name and not image:
            # one of the two has to be set
            assert False

        if image:
            blob = {}
            for m in image:
                blob[m] = image[m].copy()
        else:
            blob = self._load_data(image_name)

        if training_format:
            blob = augmentate(blob, **self.config['augmentation'])
            # transformation into one-hot-format
            blob['labels'] = np.array(np.arange(len(self.labelinfo), dtype=np.int) ==
                                      blob['labels'][:, :, None]).astype(int)
        return blob
