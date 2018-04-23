import numpy as np
from os import listdir, path, environ
import cv2
import tarfile
from sklearn.model_selection import train_test_split

from xview.settings import DATA_BASEPATH
from .data_baseclass import DataBaseclass
from .augmentation import augmentate
from copy import deepcopy


CITYSCAPES_BASEPATH = path.join(DATA_BASEPATH, 'cityscapes')

CITIES = ['aachen', 'bremen', 'darmstadt', 'erfurt', 'hanover', 'krefeld', 'strasbourg',
          'tubingen', 'weimar', 'bochum', 'cologne', 'dusseldorf', 'hamburg', 'jena',
          'monchengladbach', 'stuttgart', 'ulm', 'zurich']


class Cityscapes(DataBaseclass):

    def __init__(self, base_path=CITYSCAPES_BASEPATH, batchsize=1, in_memory=False,
                 cities=CITIES, **data_config):

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
            'resize': False
        }
        config.update(data_config)
        self.config = config

        if not path.exists(base_path):
            message = 'ERROR: Path to CITYSCAPES dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.base_path = base_path
        self.modality_paths = {
                'rgb': 'leftImg8bit_trainvaltest/leftImg8bit',
                'labels': 'gtFine_trainvaltest/gtFine',
                'depth': 'disparity_trainvaltest/disparity'
        }
        self.modality_suffixes = {
                'rgb': 'leftImg8bit',
                'labels': 'gtFine_labelIds',
                'depth': 'disparity'
        }

        original_labelinfo = {
                0: {'name': 'unlabeled', 'mapping': 'void'},
                1: {'name': 'ego vehicle', 'mapping': 'void'},
                2: {'name': 'rectification border', 'mapping': 'void'},
                3: {'name': 'out of roi', 'mapping': 'void'},
                4: {'name': 'static', 'mapping': 'void'},
                5: {'name': 'dynamic', 'mapping': 'void'},
                6: {'name': 'ground', 'mapping': 'void'},
                7: {'name': 'road', 'mapping': 'road'},
                8: {'name': 'sidewalk', 'mapping': 'sidewalk'},
                9: {'name': 'parking', 'mapping': 'road'},
                10: {'name': 'rail track', 'mapping': 'void'},
                11: {'name': 'building', 'mapping': 'building'},
                12: {'name': 'wall', 'mapping': 'building'},
                13: {'name': 'fence', 'mapping': 'fence'},
                14: {'name': 'guard rail', 'mapping': 'void'},
                15: {'name': 'bridge', 'mapping': 'void'},
                16: {'name': 'tunnel', 'mapping': 'void'},
                17: {'name': 'pole', 'mapping': 'pole'},
                18: {'name': 'polegroup', 'mapping': 'void'},
                19: {'name': 'traffic light', 'mapping': 'void'},
                20: {'name': 'traffic sign', 'mapping': 'traffic sign'},
                21: {'name': 'vegetation', 'mapping': 'vegetation'},
                22: {'name': 'terrain', 'mapping': 'vegetation'},
                23: {'name': 'sky', 'mapping': 'sky'},
                24: {'name': 'person', 'mapping': 'person'},
                25: {'name': 'rider', 'mapping': 'person'},
                26: {'name': 'car', 'mapping': 'vehicle'},
                27: {'name': 'truck', 'mapping': 'vehicle'},
                28: {'name': 'bus', 'mapping': 'vehicle'},
                29: {'name': 'caravan', 'mapping': 'vehicle'},
                30: {'name': 'trailer', 'mapping': 'vehicle'},
                31: {'name': 'train', 'mapping': 'vehicle'},
                32: {'name': 'motorcycle', 'mapping': 'vehicle'},
                33: {'name': 'bike', 'mapping': 'bicycle'}
        }

        labelinfo = {
            0: {'name': 'void', 'color': [0, 0, 0]},
            1: {'name': 'sky', 'color': [128, 128, 128]},
            2: {'name': 'building', 'color': [128, 0, 0]},
            3: {'name': 'road', 'color': [128, 64, 128]},
            4: {'name': 'sidewalk', 'color': [0, 0, 192]},
            5: {'name': 'fence', 'color': [64, 64, 128]},
            6: {'name': 'vegetation', 'color': [128, 128, 0]},
            7: {'name': 'pole', 'color': [192, 192, 128]},
            8: {'name': 'vehicle', 'color': [64, 0, 128]},
            9: {'name': 'traffic sign', 'color': [192, 128, 128]},
            10: {'name': 'person', 'color': [64, 64, 0]},
            11: {'name': 'bicycle', 'color': [0, 128, 192]}
        }

        self.label_lookup = [next(i for i in labelinfo
                                  if labelinfo[i]['name'] == k['mapping'])
                             for _, k in original_labelinfo.items()]

        # load training and test sets
        # Generate train/test splits
        def get_filenames(fileset, cities=False):
            filenames = []
            base_dir = path.join(self.base_path, self.modality_paths['rgb'], fileset)
            for city in listdir(base_dir):
                # do only include specified cities into trainset
                if cities and city not in cities:
                    continue

                search_path = path.join(base_dir, city)
                filenames.extend(
                    [{'image_name': '_'.join(path.splitext(n)[0].split('_')[:3]),
                      'image_path': path.join(fileset, city)}
                     for n in listdir(search_path)])
            return filenames

        if in_memory:
            print('INFO loading dataset into memory')
            # first load the tarfile into a closer memory location, then load all the
            # images
            tar = tarfile.open(path.join(base_path, 'cityscapes.tar.gz'))
            localtmp = environ['TMPDIR']
            tar.extractall(path=localtmp)
            tar.close()
            self.base_path = localtmp
            trainset = [{'image': self._load_data(i['image_name'], i['image_path'])}
                        for i in get_filenames('train', cities=cities)]
            testset = [{'image': self._load_data(i['image_name'], i['image_path'])}
                       for i in get_filenames('val', cities=['munster', 'frankfurt',
                                                             'lindau'])]
        else:
            trainset = get_filenames('train', cities=cities)
            testset = get_filenames('val', cities=['munster', 'frankfurt', 'lindau'])

        trainset, measureset = train_test_split(trainset, test_size=0.05,
                                                random_state=4)

        # Intitialize Baseclass
        DataBaseclass.__init__(self, trainset, measureset, testset, 14, {
            'rgb': (None, None, 3), 'depth': (None, None, 1), 'labels': (None, None)},
            labelinfo)

    def _load_data(self, image_name, image_path):
        rgb_filename, depth_filename, labels_filename = (
            path.join(self.base_path,
                      self.modality_paths[m],
                      image_path,
                      '{}_{}.png'.format(image_name,
                                         self.modality_suffixes[m]))
            for m in ['rgb', 'depth', 'labels']
        )

        blob = {}
        blob['rgb'] = cv2.imread(rgb_filename)
        blob['depth'] = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
        blob['labels'] = cv2.imread(labels_filename, cv2.IMREAD_ANYDEPTH)
        # apply label mapping
        blob['labels'] = np.asarray(self.label_lookup, dtype='int32')[blob['labels']]

        if self.config['resize']:
            blob['rgb'] = cv2.resize(blob['rgb'], (768, 384),
                                     interpolation=cv2.INTER_LINEAR)
            for m in ['depth', 'labels']:
                blob[m] = cv2.resize(blob[m], (768, 384),
                                     interpolation=cv2.INTER_NEAREST)

        blob['depth'] = np.expand_dims(blob['depth'], 3)
        return blob

    def _get_data(self, image_name=False, image_path=False, image=False,
                  training_format=False):
        """Returns data for one given image number from the specified sequence."""
        if not image_name and not image:
            # one of the two has to be set
            assert False

        if image:
            blob = {}
            for m in image:
                blob[m] = image[m].copy()
        else:
            blob = self._load_data(image_name, image_path)

        if training_format:
            blob = augmentate(blob, **self.config['augmentation'])
            # transformation into one-hot-format
            blob['labels'] = np.array(np.arange(len(self.labelinfo), dtype=np.int) ==
                                      blob['labels'][:, :, None]).astype(int)
        return blob

    def get_ego_vehicle_mask(self, image_name, image_path):
        # save the old label mapping
        old_label_lookup = deepcopy(self.label_lookup)

        # now map everything on 0 except for ego-vehicle, which is mapped on 1
        self.label_lookup = [0 for _ in range(34)]
        self.label_lookup[1] = 1

        blob = self._load_data(image_name, image_path)

        # restore labellookup
        self.label_lookup = old_label_lookup
        return blob
