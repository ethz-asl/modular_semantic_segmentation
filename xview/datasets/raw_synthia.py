import numpy as np
from os import listdir, path, makedirs
from tqdm import tqdm
import cv2
import shutil
import json
import random

from .data_baseclass import DataBaseclass
from .synthia import SYNTHIA_BASEPATH, AVAILABLE_SEQUENCES, LABELINFO, \
    one_channel_image_reader, one_hot_lookup


class Synthia(DataBaseclass):
    """Driver for SYNTHIA dataset (http://synthia-dataset.net/).
    Preprocessing resizes images to 640x368 and performs a static 20% test-split for all
    given sequences."""

    def __init__(self, base_path=SYNTHIA_BASEPATH, force_preprocessing=False,
                 batchsize=1, **data_config):

        config = {
            'seqs': AVAILABLE_SEQUENCES,
            'direction': 'F',
            'augmentation': {
                'crop': 480,
                'scale': [0.7, 1.5],
                'vflip': True,
                'hflip': False,
                'gamma': [0.3, 2]
            }
        }
        config.update(data_config)
        self.config = config

        if not path.exists(base_path):
            message = 'ERROR: Path to SYNTHIA dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        if not len(config['seqs']) > 0:
            print('ERROR: Need to specify at least one synthia set')
            raise UserWarning('ERROR: Need to specify at least one synthia set')

        self.base_path = base_path

        # For some sequences, preprocessing may be necessary.
        for sequence in config['seqs']:
            if force_preprocessing or \
                    not path.exists(path.join(base_path,
                                              '{}/GT/LABELS_NPY'.format(sequence))):
                self._preprocessing(sequence)

        # Every sequence got their own train/test split during preprocessing. According
        # to the loaded sequences, we now collect all files from all sequence-subsets
        # into one list.
        trainset = []
        testset = []
        for sequence in config['seqs']:
            train_test_file = path.join(self.base_path,
                                        '{}/train_test_split.json'.format(sequence))
            with open(train_test_file, 'r') as f:
                split = json.load(f)
                trainset.extend([{'sequence': sequence, 'image_name': filename}
                                 for filename in split['trainset']])
                testset.extend([{'sequence': sequence, 'image_name': filename}
                                for filename in split['testset']])

        # Intitialize Baseclass
        DataBaseclass.__init__(self, trainset, testset, batchsize,
                               ['rgb', 'depth', 'labels'], LABELINFO)

    def _preprocessing(self, sequence):
        rootpath = path.join(self.base_path, sequence, 'GT')

        for direction in ['F', 'B', 'L', 'R']:
            inpath, outpath = (path.join(rootpath, pref,
                                         'Stereo_Right/Omni_{}'.format(direction))
                               for pref in ['LABELS', 'LABELS_NPY'])

            if path.exists(outpath):
                shutil.rmtree(outpath)
            makedirs(outpath)
            for filename in tqdm(listdir(inpath)):
                array = one_channel_image_reader(path.join(inpath, filename),
                                                 np.uint8)
                np.save(path.join(outpath, filename.split('.')[0]), array)

    def _get_data(self, sequence, image_name, training_format=True):
        """Returns data for one given image number from the specified sequence."""
        filetype = {'rgb': 'png', 'depth': 'png', 'labels': 'npy'}

        rgb_filename, depth_filename, groundtruth_filename = (
            path.join(self.base_path, '{}/{}/Stereo_Right/Omni_{}/{}.{}'
                      .format(sequence, pref, self.config['direction'],
                              image_name, filetype[modality]))
            for pref, modality in zip(['RGB', 'Depth', 'GT/LABELS_NPY'],
                                      ['rgb', 'depth', 'labels']))

        blob = {}
        blob['rgb'] = cv2.imread(rgb_filename)
        depth = one_channel_image_reader(depth_filename.format('png'), np.uint16,
                                         input_has_three_channels=False)
        # We have to add a dimension for the channels, as there is only one and the
        # dimension is omitted.
        blob['depth'] = np.expand_dims(depth, 3)
        labels = np.load(groundtruth_filename)
        # Dirty fix for the class 15
        labels[labels == 15] = 13
        blob['labels'] = labels

        if training_format:
            scale = self.config['augmentation']['scale']
            crop = self.config['augmentation']['crop']
            hflip = self.config['augmentation']['hflip']
            vflip = self.config['augmentation']['vflip']
            gamma = self.config['augmentation']['gamma']

            if scale and crop:
                h, w, _ = blob['rgb'].shape
                min_scale = crop / float(min(h, w))
                k = random.uniform(max(min_scale, scale[0]), scale[1])
                blob['rgb'] = cv2.resize(blob['rgb'], None, fx=k, fy=k)
                blob['depth'] = cv2.resize(blob['depth'], None, fx=k, fy=k)
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

            # Format labels into one-hot
            blob['labels'] = np.array(one_hot_lookup ==
                                      blob['labels'][:, :, None]).astype(int)

        # Force the image dimension to be multiple of 16
        h, w, _ = blob['rgb'].shape
        h_c, w_c = [d - (d % 16) for d in [h, w]]
        if h_c != h or w_c != w:
            for m in ['rgb', 'depth', 'labels']:
                blob[m] = blob[m][:h_c, :w_c, ...]

        return blob
