import numpy as np
from os import listdir, path
import cv2
import random

from xview.settings import DATA_BASEPATH
from .data_baseclass import DataBaseclass


CITYSCAPES_BASEPATH = path.join(DATA_BASEPATH, 'cityscapes')


class Cityscapes(DataBaseclass):

    def __init__(self, base_path=CITYSCAPES_BASEPATH, **data_config):

        config = {
            'batchsize': 1,
            'preprocessing': {'type': 'offline'}
        }
        config.update(data_config)

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

        # Generate train/test splits
        trainset = []
        testset = []
        for s, set_name in zip([trainset, testset], ['train', 'val']):
            base_dir = path.join(base_path, self.modality_paths['rgb'], set_name)
            for city in listdir(base_dir):
                search_path = path.join(base_dir, city)
                s.extend(
                    [{'image_name': '_'.join(path.splitext(n)[0].split('_')[:3]),
                      'image_path': path.join(set_name, city)}
                     for n in listdir(search_path)]
                )

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
                19: {'name': 'traffic light', 'mapping': 'traffic light'},
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
            11: {'name': 'bicycle', 'color': [0, 128, 192]},
            12: {'name': 'traffic light', 'color': [0, 128, 128]}
        }

        self.label_lookup = [(i for i in labelinfo
                              if labelinfo[i]['name'] == k['mapping']).next()
                             for _, k in original_labelinfo.iteritems()]

        # Intitialize Baseclass
        DataBaseclass.__init__(self, trainset, testset, ['rgb', 'depth', 'labels'],
                               labelinfo, **config)

    @property
    def one_hot_lookup(self):
        return np.arange(len(self.labelinfo), dtype=np.int)

    def _get_data(self, image_name, image_path, one_hot=True, preproc_type=-1):
        """Returns data for one given image number from the specified sequence."""
        preproc_type = self.config['preprocessing']['type'] \
            if preproc_type is -1 else preproc_type
        filetype = {'rgb': 'png', 'depth': 'png', 'labels': 'png'}

        rgb_filename, depth_filename, labels_filename = (
            path.join(self.base_path,
                      self.modality_paths[m],
                      image_path,
                      '{}_{}.{}'.format(image_name,
                                        self.modality_suffixes[m],
                                        filetype[m]))
            for m in ['rgb', 'depth', 'labels']
        )

        blob = {}
        blob['rgb'] = cv2.imread(rgb_filename)
        blob['depth'] = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
        blob['labels'] = cv2.imread(labels_filename, cv2.IMREAD_ANYDEPTH)

        if preproc_type == 'online':
            scale = self.config['preprocessing'].get('scale')
            crop  = self.config['preprocessing'].get('crop')
            hflip = self.config['preprocessing'].get('hflip')
            vflip = self.config['preprocessing'].get('vflip')
            gamma = self.config['preprocessing'].get('gamma')

            if scale and crop:
                h, w, _ = blob['rgb'].shape
                min_scale = crop / float(min(h, w))
                k = random.uniform(max(min_scale, scale[0]), scale[1])
                blob['rgb'] = cv2.resize(blob['rgb'], None, fx=k, fy=k)
                blob['depth'] = cv2.resize(blob['depth'], None, fx=k, fy=k,
                                           interpolation=cv2.INTER_NEAREST)
                blob['labels'] = cv2.resize(blob['labels'], None, fx=k, fy=k,
                                            interpolation=cv2.INTER_NEAREST)

            if crop:
                h, w, _ = blob['rgb'].shape
                h_c = random.randint(0, h - crop)
                w_c = random.randint(0, w - crop)
                for m in ['rgb', 'depth', 'labels']:
                    blob[m] = blob[m][h_c:h_c+crop, w_c:w_c+crop, ...]

            if hflip and np.random.choice([0, 1]):
                for m in ['rgb', 'depth', 'labels']:
                    blob[m] = np.flip(blob[m], axis=0)

            if vflip and np.random.choice([0, 1]):
                for m in ['rgb', 'depth', 'labels']:
                    blob[m] = np.flip(blob[m], axis=1)

            if gamma:
                k = random.uniform(gamma[0], gamma[1])
                lut = np.array([((i / 255.0) ** (1/k)) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
                blob['rgb'] = lut[blob['rgb']]

        force_multiple = self.config['preprocessing'].get('force_multiple')
        if force_multiple:
            h, w, _ = blob['rgb'].shape
            h_c, w_c = [d - (d % force_multiple) for d in [h, w]]
            if h_c != h or w_c != w:
                for m in ['rgb', 'depth', 'labels']:
                    blob[m] = blob[m][:h_c, :w_c, ...]

        blob['depth'] = np.expand_dims(blob['depth'], 3)
        blob['labels'] = np.asarray(self.label_lookup)[blob['labels']]

        if one_hot:
            blob['labels'] = np.array(self.one_hot_lookup ==
                                      blob['labels'][:, :, None]).astype(int)
        return blob
