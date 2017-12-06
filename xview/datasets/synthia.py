import numpy as np
from os import listdir, path, mkdir
from scipy.misc import imread
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from png import Reader, Writer
from PIL import Image
import cv2
import itertools
import shutil
import json

from xview.settings import DATA_BASEPATH
from xview.datasets.data_baseclass import DataBaseclass


SYNTHIA_BASEPATH = path.join(DATA_BASEPATH, 'synthia')

AVAILABLE_SEQUENCES = ['SYNTHIA-SEQS-04-DAWN',
                       'SYNTHIA-SEQS-04-FALL',
                       'SYNTHIA-SEQS-04-FOG',
                       'SYNTHIA-SEQS-04-NIGHT',
                       'SYNTHIA-SEQS-04-RAINNIGHT',
                       'SYNTHIA-SEQS-04-SOFTRAIN',
                       'SYNTHIA-SEQS-04-SPRING',
                       'SYNTHIA-SEQS-04-SUMMER',
                       'SYNTHIA-SEQS-04-SUNSET',
                       'SYNTHIA-SEQS-04-WINTER',
                       'SYNTHIA-SEQS-04-WINTERNIGHT']

# Set label information according to synthia README
LABELINFO = {
   0: {'name': 'void', 'color': [0, 0, 0]},
   1: {'name': 'sky', 'color': [128, 128, 128]},
   2: {'name': 'building', 'color': [128, 0, 0]},
   3: {'name': 'road', 'color': [128, 64, 128]},
   4: {'name': 'sidewalk', 'color': [0, 0, 192]},
   5: {'name': 'fence', 'color': [64, 64, 128]},
   6: {'name': 'vegetation', 'color': [128, 128, 0]},
   7: {'name': 'pole', 'color': [192, 192, 128]},
   8: {'name': 'car', 'color': [64, 0, 128]},
   9: {'name': 'traffic sign', 'color': [192, 128, 128]},
   10: {'name': 'pedestrian', 'color': [64, 64, 0]},
   11: {'name': 'bicycle', 'color': [0, 128, 192]},
   12: {'name': 'lanemarking', 'color': [0, 192, 0]},
   13: {'name': 'traffic light', 'color': [0, 128, 128]}
}

"""For some reason, synthia consists of the labels 0-12 and 15. to create a
        onw-hot vector of these 14 classes, one can compare agains the following array,
            e.g. the one-hot version of class 4 is:
                (self.one_hot_lookup == 4).astype(int)
                  -->     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0]"""
one_hot_lookup = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])


class Synthia(DataBaseclass):
    """Driver for SYNTHIA dataset (http://synthia-dataset.net/).
    Preprocessing resizes images to 640x368 and performs a static 20% test-split for all
    given sequences."""

    def __init__(self, batchsize, seqs=AVAILABLE_SEQUENCES, base_path=SYNTHIA_BASEPATH,
                 force_preprocessing=False, direction='F'):
        if not path.exists(base_path):
            message = 'ERROR: Path to SYNTHIA dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        if not len(seqs) > 0:
            print('ERROR: Need to specify at least one synthia set')
            raise UserWarning('ERROR: Need to specify at least one synthia set')

        self.base_path = base_path

        # For some sequences, preprocessing may be necessary.
        for sequence in seqs:
            if force_preprocessing or \
                    not path.exists(path.join(base_path,
                                              '{}/resized_rgb_F'.format(sequence))):
                self._preprocessing(sequence)

        # Every sequence got their own train/test split during preprocessing. According
        # to the loaded sequences, we now collect all files from all sequence-subsets
        # into one list.
        trainset = []
        testset = []
        for sequence in seqs:
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
        # Save direction
        self.direction = direction

    def _preprocessing(self, sequence):
        """Preprocessing of SYNTHIA data.

        Performs several steps:
            - Original images are of size 1280x760. To make training and data loading
              faster, we resize them to a width of 640 (factor 2). However, picture
              dimensions need to be a multiple of 16 to pass through 4 levels of
              pooling in VGG16, therefore we have to crop the image a little bit in
              height and end up at a size of 640x368
            - Resizing is done with boilinear interpolation for RGB and taking the upper-
              left corner of every group of pixels for depth and label.
            - Labels are stored in a crude version of the png format. We open this,
              extract the label integer from the first channel (we omit the item ID as we
              do not use it) and store it as numpy file.
            - From the available images, we select a random 20% sample as a test-set and
              store the indexes of the train/test split so they are consistent between
              training runs.
        """
        print('INFO: Preprocessing started for {}. This may take a while.'
              .format(sequence))
        sequence_basepath = path.join(self.base_path, sequence)

        # First we create directories for the downsamples images.
        for modality, direction in itertools.product(['RGB', 'Depth', 'labels'],
                                                     ['F', 'B', 'L', 'R']):
            new_images = path.join(sequence_basepath,
                                   'resized_{}_{}'.format(modality.lower(), direction))
            if path.exists(new_images):
                # We are doing a fresh preprocessing, so delete old data.
                shutil.rmtree(new_images)
            mkdir(new_images)

            # RGB and Depth Images can simply be resized by PIL.
            if modality in ['RGB', 'Depth']:
                original_images = path.join(sequence_basepath, '{}/Stereo_Right/Omni_{}'
                                            .format(modality, direction))

                # As we have to handle different bitdepths and cropped images, we use
                # pypng for writing the new files.
                bitdepth = 8 if modality == 'RGB' else 16
                writer = Writer(width=640, height=368, bitdepth=bitdepth,
                                greyscale=(modality == 'Depth'))

                for filename in listdir(original_images):
                    if modality == 'RGB':
                        image = Image.open(path.join(original_images, filename))
                        resized = image.resize([640, 380], resample=Image.BILINEAR)
                        resized = np.asarray(resized, dtype=np.uint8)
                    elif modality == 'Depth':
                        image = one_channel_image_reader(
                            path.join(original_images, filename), np.uint16)
                        resized = zoom(image, 0.5, mode='nearest', order=0)
                    # The image has to get cropped, see docstring of crop method
                    cropped = crop_resized_image(resized)
                    # Now save again accoding to different formats
                    with open(path.join(new_images, filename), 'wb') as f:
                        # The writer is expecting a consecutive list of RGBRGBR... values
                        if modality == 'RGB':
                            val_list = cropped.reshape(368, 640*3).tolist()
                        elif modality == 'Depth':
                            val_list = cropped.tolist()
                        writer.write(f, val_list)
                continue

            # Label image has a weird two-channel format where the 1st channel is the
            # 8bit label integer and the second channel is the instance ID we do not use.
            # We save the extracted label integer as numpy array to make things easier.
            original_images = path.join(
                sequence_basepath, 'GT/LABELS/Stereo_Right/Omni_{}'.format(direction))

            for filename in listdir(original_images):
                # Labels are stored in the 1st channel of the png file
                array = one_channel_image_reader(path.join(original_images, filename),
                                                 np.uint8)
                resized = zoom(array, 0.5, mode='nearest', order=0)
                # The image has to get cropped, se docstring of crop method
                cropped = crop_resized_image(resized)
                # Now we can save it as a numpy array
                np.save(path.join(new_images, filename.split('.')[0]), cropped)

        filenames = [filename.split('.')[0] for filename
                     in listdir(path.join(sequence_basepath, 'resized_rgb_F'))]
        trainset, testset = train_test_split(filenames, test_size=0.2)
        with open('{}/train_test_split.json'.format(sequence_basepath), 'w') as f:
            json.dump({'trainset': trainset, 'testset': testset}, f)
        print('INFO: Preprocessing finished.')

    def _get_data(self, sequence, image_name, training_format=True):
        """Returns data for one given image number from the specified sequence."""
        filetype = {'rgb': 'png', 'depth': 'png', 'labels': 'npy'}
        rgb_filename, depth_filename, groundtruth_filename = (
            path.join(self.base_path, '{}/resized_{}_{}/{}.{}'
                      .format(sequence, modality, self.direction, image_name,
                              filetype[modality]))
            for modality in ['rgb', 'depth', 'labels'])

        blob = {}
        blob['rgb'] = cv2.imread(rgb_filename.format('.png'))
        depth = cv2.imread(depth_filename.format('png'), 2)
        # We have to add a dimension for the channels, as there is only one and the
        # dimension is omitted
        blob['depth'] = np.expand_dims(depth, 3)
        labels = np.load(groundtruth_filename.format('.npy'))
        # Dirty fix for the class 15
        labels[labels == 15] = 13
        if training_format:
            # Labels still have to get converted to one-hot
            labels = np.array(one_hot_lookup == labels[:, :, None]).astype(int)
        blob['labels'] = labels
        return blob


def one_channel_image_reader(filepath, datatype, input_has_three_channels=True):
    """Labels are stored in a crude way into the png format that cannot be handled by
    standard libraries. Therefore, we have to create this decoder."""
    im = Reader(filepath)
    _, _, array, _ = im.asDirect()
    array = np.vstack(itertools.imap(datatype, array))
    if input_has_three_channels:
        # This image array is now in 'boxed row flat pixel' format, meaning that each row
        # is just a continuous list of R, G, B, R, G, B, R, ... values.
        # We are in this case only interested in the first component, which holds the
        # class label, therefore we take every third value.
        array = array[:, range(0, array.shape[1], 3)]
    return datatype(array)


def crop_resized_image(image):
    """We resize the images to half of their original size, 640x380. However, in order
    to use the image with VGG16, each dimension should be devisible by 16 as we have 4
    pooling layers. We therefore need to crop the images in a consistant way to the size
    of 640x368."""
    return image[6:374]
