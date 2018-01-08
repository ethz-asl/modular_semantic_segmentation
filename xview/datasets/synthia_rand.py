import numpy as np
from os import path
import cv2
import json
import random

from xview.settings import DATA_BASEPATH
from .baseclass import DataBaseclass


SYNTHIA_BASEPATH = path.join(DATA_BASEPATH, 'synthia_rand')


class SynthiaRand(DataBaseclass):

    def __init__(self, base_path=SYNTHIA_BASEPATH, **data_config):

        config = {
            'direction': 'F',
            'batchsize': 1,
            'preprocessing': {'type': 'offline'}
        }
        config.update(data_config)

        if not path.exists(base_path):
            message = 'ERROR: Path to SYNTHIA-RAND dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.base_path = base_path

        train_test_file = path.join(self.base_path, 'train_test_split.json')
        with open(train_test_file, 'r') as f:
            split = json.load(f)
        trainset = [{'image_name': filename} for filename in split['trainset']]
        testset = [{'image_name': filename} for filename in split['testset']]

        original_labelinfo = {
                0: {'name': 'void', 'mapping': 'void'},
                1: {'name': 'sky', 'mapping': 'sky'},
                2: {'name': 'building', 'mapping': 'building'},
                3: {'name': 'road', 'mapping': 'road'},
                4: {'name': 'sidewalk', 'mapping': 'sidewalk'},
                5: {'name': 'fence', 'mapping': 'fence'},
                6: {'name': 'vegetation', 'mapping': 'vegetation'},
                7: {'name': 'pole', 'mapping': 'pole'},
                8: {'name': 'car', 'mapping': 'vehicle'},
                9: {'name': 'traffic sign', 'mapping': 'traffic sign'},
                10: {'name': 'pedestrian', 'mapping': 'person'},
                11: {'name': 'bicycle', 'mapping': 'bicycle'},
                12: {'name': 'motorcycle', 'mapping': 'vehicle'},
                13: {'name': 'parking slot', 'mapping': 'road'},
                14: {'name': 'road work', 'mapping': 'void'},
                15: {'name': 'traffic light', 'mapping': 'traffic light'},
                16: {'name': 'terrain', 'mapping': 'vegetation'},
                17: {'name': 'rider', 'mapping': 'person'},
                18: {'name': 'truck', 'mapping': 'vehicle'},
                19: {'name': 'bus', 'mapping': 'vehicle'},
                20: {'name': 'train', 'mapping': 'vehicle'},
                21: {'name': 'wall', 'mapping': 'building'},
                22: {'name': 'lanemarking', 'mapping': 'road'},
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

    def _get_data(self, image_name, one_hot=True, preproc_type=-1):
        """Returns data for one given image number from the specified sequence."""
        preproc_type = self.config['preprocessing']['type'] \
            if preproc_type is -1 else preproc_type

        filetype = {'rgb': 'png', 'depth': 'png', 'labels': 'npy'}
        rgb_filename, depth_filename, groundtruth_filename = (
            path.join(self.base_path, '{}/Stereo_Right/Omni_{}/{}.{}'
                      .format(pref, self.config['direction'],
                              image_name, filetype[modality]))
            for pref, modality in zip(['RGB', 'Depth', 'GT/LABELS_NPY'],
                                      ['rgb', 'depth', 'labels']))

        blob = {}
        blob['rgb'] = cv2.imread(rgb_filename)
        blob['depth'] = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
        blob['labels'] = np.load(groundtruth_filename)

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
