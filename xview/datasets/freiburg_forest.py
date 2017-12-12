import numpy as np
from scipy.misc import imread
from os import listdir, path, mkdir
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import zoom
import tifffile as tiff
import random
import cv2

from xview.settings import DATA_BASEPATH
from .data_baseclass import DataBaseclass, crop_multiple
from .synthia import one_channel_image_reader


FOREST_BASEPATH = path.join(DATA_BASEPATH, 'freiburg_forest')

LABELINFO = {
    0: {'name': 'void', 'color': [255, 255, 255]},
    1: {'name': 'road', 'color': [170, 170, 170]},
    2: {'name': 'grass', 'color': [0, 255, 0]},
    3: {'name': 'vegetation', 'color': [102, 102, 51]},
    4: {'name': 'sky', 'color': [0, 120, 255]},
    5: {'name': 'obstacle', 'color': [0, 0, 0]},
}


class FreiburgForest(DataBaseclass):

    def __init__(self, base_path=FOREST_BASEPATH, batchsize=1,
                 force_preprocessing=False, **config):

        if not path.exists(base_path):
            message = 'ERROR: Path to SYNTHIA dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.base_path = base_path

        # Every sequence got their own train/test split during preprocessing. According
        # to the loaded sequences, we now collect all files from all sequence-subsets
        # into one list.
        trainset, testset = (
            [{'fileset': fileset, 'image_name': filename.split('.')[0].split('_')[0]}
             for filename in listdir(path.join(self.base_path, fileset, 'GT_color'))]
            for fileset in ['train', 'test'])

        if force_preprocessing:
            self._preprocessing(trainset + testset)

        self.config = config

        # Intitialize Baseclass
        DataBaseclass.__init__(self, trainset, testset, batchsize,
                               ['rgb', 'depth', 'labels', 'evi', 'ndvi', 'nir', 'nrg'],
                               LABELINFO)

    def _preprocessing(self, image_list):
        modality_paths = {'rgb': 'rgb', 'depth': 'depth_gray', 'labels': 'GT_color',
                          'evi': 'evi_gray', 'ndvi': 'ndvi_float', 'nir': 'nir',
                          'nrg': 'nrg'}

        for descriptor in tqdm(image_list, desc='Files'):
            # Is it train or test file? which image ID?
            fileset, image_name = descriptor['fileset'], descriptor['image_name']
            for modality, data_path in modality_paths.items():
                directory = path.join(self.base_path, fileset, data_path)
                filename = (file for file in listdir(directory)
                            if file.startswith(image_name)).next()
                filepath = path.join(directory, filename)

                # We resize every image to a width of 600px (depth is only given in this
                # format) and then crop the height to 300px. Afterwards, we crop
                # everything to match a multiple of 16.

                if modality in ['rgb', 'evi', 'nir', 'nrg', 'ndvi']:
                    image = Image.open(filepath)
                    width, height = image.size
                    new_height = int(height * 600.0 / width + 0.5)
                    resized = image.resize([600, new_height], resample=Image.BILINEAR)
                    if modality == 'ndvi':
                        resized = np.asarray(resized, dtype='float32')
                    else:
                        resized = np.asarray(resized, dtype='uint8')
                elif modality == 'labels':
                    # LABELS get resized with nearest-neighbour method
                    # The label is encoded as color and has to get looked up in the
                    # decision-functions defined below.
                    color_labels = imread(filepath)

                    color_sum = color_labels.sum(2)

                    labels = -1 * np.ones_like(color_sum)
                    labels[color_sum == 510] = 1
                    labels[np.logical_and(color_sum == 255,
                                          color_labels[..., 0] == 0)] = 2
                    labels[np.logical_and(color_sum == 255,
                                          color_labels[..., 0] == 102)] = 3
                    labels[color_sum == 60] = 3
                    labels[color_sum == 375] = 4
                    labels[color_sum == 0] = 5

                    assert np.sum(labels == -1) == 0

                    zoom_factor = 600.0 / labels.shape[1]
                    resized = zoom(labels, zoom_factor, mode='nearest', order=0)

                else:
                    # DEPTH is already resized
                    resized = one_channel_image_reader(filepath, np.uint16,
                                                       input_has_three_channels=False)

                # Now crop the resized data to a height of 267 (smallest found)
                current_height = resized.shape[0]
                lower_crop = int((current_height - 267) / 2.0 + 0.5)
                resized = resized[lower_crop:(lower_crop + 267), :]

                # Force the image dimension to be multiple of 16
                resized = crop_multiple(resized)

                # Now save the image
                new_filepath = path.join(self.base_path, fileset,
                                         'resized_{}'.format(modality))

                if not path.exists(new_filepath):
                    mkdir(new_filepath)

                if modality == 'labels':
                    np.save(path.join(new_filepath, '{}.npy'.format(image_name)),
                            resized)
                else:
                    try:
                        tiff.imsave(path.join(new_filepath, '{}.tif'.format(image_name)),
                                    resized)
                    except ValueError as ve:
                        print(image_name, modality)
                        raise ve

    def _get_data(self, fileset, image_name, training_format=True):
        """Returns data for one given image number from the specified sequence."""
        blob = {}
        for modality in self.modalities:
            # image_name is in format (train/test)_ID. Therefore, the first part tells
            # whether the image is located in train- or test-directory and the second the
            # prefix of the filename.
            directory = path.join(self.base_path, fileset, 'resized_{}'.format(modality))
            filename = (file for file in listdir(directory)
                        if file.startswith(image_name)).next()
            filepath = path.join(directory, filename)

            if modality == 'labels':
                labels = np.load(filepath)

                if training_format:
                    # Convert the labels into one-hot encoding.
                    blob['labels'] = np.array(np.array(list(range(6))) ==
                                              labels[:, :, None]).astype('int')
                else:
                    blob['labels'] = labels
            else:
                blob[modality] = imread(filepath)

        if training_format:
            scale = self.config['augmentation']['scale']
            crop = self.config['augmentation']['crop']
            hflip = self.config['augmentation']['hflip']
            vflip = self.config['augmentation']['vflip']

            if scale and crop:
                h, w, _ = blob['rgb'].shape
                min_scale = crop / float(min(h, w))
                k = random.uniform(max(min_scale, scale[0]), scale[1])
                for modality in self.modalities:
                    if modality == 'rgb':
                        blob['rgb'] = cv2.resize(blob['rgb'], None, fx=k, fy=k)
                    else:
                        blob[modality] = cv2.resize(blob[modality], None, fx=k, fy=k,
                                                    interpolation=cv2.INTER_NEAREST)

            if crop:
                h, w, _ = blob['rgb'].shape
                h_c = random.randint(0, h - crop)
                w_c = random.randint(0, w - crop)
                for m in self.modalities:
                    blob[m] = blob[m][h_c:h_c+crop, w_c:w_c+crop, ...]

            if hflip and np.random.choice([0, 1]):
                for m in self.modalities:
                    blob[m] = np.flip(blob[m], axis=0)

            if vflip and np.random.choice([0, 1]):
                for m in self.modalities:
                    blob[m] = np.flip(blob[m], axis=1)

        for modality in self.modalities:
            # We have to add a dimension for the channels, as there is only one and the
            # dimension is omitted.
            if len(blob[modality].shape) < 3 and modality != 'labels':
                blob[modality] = np.expand_dims(blob[modality], 3)

        return blob
