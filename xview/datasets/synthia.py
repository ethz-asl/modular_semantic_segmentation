import numpy as np
from os import listdir, path, mkdir
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
from png import Reader
from PIL import Image
import itertools
import shutil
import json

from xview import DATA_BASEPATH
from .baseclass import DataBaseclass


SYNTHIA_BASEPATH = path.join(DATA_BASEPATH, 'synthia')


class Synthia(DataBaseclass):

    def __init__(self, seqs, batchsize, base_path=SYNTHIA_BASEPATH,
                 force_preprocessing=False):
        if not path.exists(base_path):
            message = 'ERROR: Path to SYNTHIA dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        if not len(seqs) > 0:
            print('ERROR: Need to specify at least one synthia set')
            raise UserWarning('ERROR: Need to specify at least one synthia set')

        self.base_path = base_path

        for sequence in seqs:
            if force_preprocessing or not path.exists(path.join(base_path,
                    '{}/resized_rgb'.format(sequence))):
                self._preprocessing(sequence)

        # Every sequence got their own train/test split during preprocessing. According
        # to the loaded sequences, we now collect all files from all sequence-subsets
        # into one list.
        trainset = []
        testset = []
        for sequence in seqs:
            train_test_names = path.join(self.base_path,
                                         '{}/train_test_split.json'.format(sequence))
            print(train_test_names)
            with open(train_test_names, 'r') as f:
                split = json.load(f)
                trainset.extend([{'sequence': sequence, 'image_name': filename}
                                 for filename in split['trainset']])
                testset.extend([{'sequence': sequence, 'image_name': filename}
                                for filename in split['testset']])

        # Intitialize Baseclass
        DataBaseclass.__init__(self, trainset, testset, batchsize)

    def _preprocessing(self, sequence):
        print('INFO: Preprocessing started for SYNTHIA Dataset. This may take a while.')
        sequence_basepath = path.join(self.base_path, sequence)
        # Dependent on iamge type, we have to use different resize filters.
        interpolation_method_for_modality = {
            'RGB': Image.BILINEAR,
            'Depth': Image.NEAREST,
            'labels': 'nearest'
        }
        # First we create directories for the downsamples images.
        for modality in ['RGB', 'Depth', 'labels']:
            new_images = path.join(sequence_basepath,
                                   'resized_{}'.format(modality.lower()))
            if path.exists(new_images):
                # We are doing a fresh preprocessing, so delete old data.
                shutil.rmtree(new_images)
            mkdir(new_images)

            # RGB and Depth Images can simply be resized by PIL.
            if modality in ['RGB', 'Depth']:
                original_images = path.join(sequence_basepath, '{}/Stereo_Right/Omni_F'
                                            .format(modality))
                for filename in listdir(original_images):
                    image = Image.open(path.join(original_images, filename))
                    resized = image.resize([640, 380],
                        resample=interpolation_method_for_modality[modality])
                    resized.save(path.join(new_images, filename))
                continue

            # Label image format cannot be decoded by standard libraries. We save the
            # extracted information as numpy array.
            original_images = path.join(sequence_basepath,
                                        'GT/LABELS/Stereo_Right/Omni_F')
            for filename in listdir(original_images):
                array = decode_labels(path.join(original_images, filename))
                resized = imresize(array, [380, 640],
                                   interpolation_method_for_modality['labels'])
                np.save(path.join(new_images, filename.split('.')[0]),
                        np.array(resized))

        filenames = [filename.split('.')[0]
                     for filename in listdir(path.join(sequence_basepath, 'resized_rgb'))]
        trainset, testset = train_test_split(filenames, test_size=0.2)
        with open('{}/train_test_split.json'.format(sequence_basepath), 'w') as f:
            json.dump({'trainset': trainset, 'testset': testset}, f)
        print('INFO: Preprocessing finished.')

    def _get_data(self, sequence, image_name):
        filetype = {'rgb': 'png', 'depth': 'png', 'labels': 'npy'}
        rgb_filename, depth_filename, groundtruth_filename = (
            path.join(self.base_path, '{}/resized_{}/{}.{}'
                      .format(sequence, modality, image_name, filetype[modality]))
            for modality in ['rgb', 'depth', 'labels'])

        blob = {}
        blob['rgb'] = imread(rgb_filename.format('.png'))
        blob['depth'] = imread(depth_filename.format('png'))
        blob['labels'] = np.load(groundtruth_filename.format('.npy'))
        return blob


def decode_labels(filepath):
    """Labels are stored in a crude way into the png format that cannot be handled by
    standard libraries. Therefore, we have to create this decoder."""
    im = Reader(filepath)
    _, _, array, _ = im.asDirect()
    array = np.vstack(itertools.imap(np.uint8, array))
    # This image array is now in 'boxed row flat pixel' format, meaning that each row
    # is just a continuous list of R, G, B, R, G, B, R, ... values.
    # We are in this case only interested in the first component, which holds the
    # class label, therefore we take every third value.
    array = array[:, range(0, 3 * 1280, 3)]
    return np.uint8(array)
