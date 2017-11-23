import numpy as np
from scipy.misc import imread
from os import listdir, path

from xview.settings import DATA_BASEPATH
from .data_baseclass import DataBaseclass
from .synthia import one_channel_image_reader


FOREST_BASEPATH = path.join(DATA_BASEPATH, 'freiburg_forest')

LABELINFO = {
    0: {'name': 'void', 'color': [255, 255, 255]},
    1: {'name': 'road', 'color': [170, 170, 170]},
    2: {'name': 'grass', 'color': [0, 255, 0]},
    3: {'name': 'vegetation', 'color': [102, 102, 51]},
    4: {'name': 'tree', 'color': [0, 60, 0]},
    5: {'name': 'sky', 'color': [0, 120, 255]},
    6: {'name': 'obstacle', 'color': [0, 0, 0]},
}


class FreiburgForest(DataBaseclass):

    def __init__(self, base_path=FOREST_BASEPATH, batchsize=1, **config):

        if not path.exists(base_path):
            message = 'ERROR: Path to SYNTHIA dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.base_path = base_path

        # Every sequence got their own train/test split during preprocessing. According
        # to the loaded sequences, we now collect all files from all sequence-subsets
        # into one list.
        trainset, testset = (
            [{'fileset': fileset, 'image_name': filename.split('-')[0]}
             for filename in listdir(path.join(self.base_path, fileset, 'GT_color'))]
            for fileset in ['train', 'test'])

        # Intitialize Baseclass
        DataBaseclass.__init__(self, trainset, testset, batchsize,
                               ['rgb', 'depth', 'labels', 'evi', 'ndvi', 'nir', 'nrg'],
                               LABELINFO, **config)

    def _get_data(self, fileset, image_name, training_format=True):
        """Returns data for one given image number from the specified sequence."""
        modality_paths = {'rgb': 'rgb', 'depth': 'depth_gray', 'labels': 'GT_color',
                          'evi': 'evi_gray', 'ndvi': 'ndvi_float', 'nir': 'nir',
                          'nrg': 'nrg'}
        blob = {}
        for modality, data_path in modality_paths.items():
            print(modality, data_path, image_name)
            # image_name is in format (train/test)_ID. Therefore, the first part tells
            # whether the image is located in train- or test-directory and the second the
            # prefix of the filename.
            directory = path.join(self.base_path, fileset, data_path)
            filename = (file for file in listdir(directory)
                        if file.startswith(image_name)).next()
            filepath = path.join(directory, filename)

            if modality in ['rgb', 'evi', 'nrg', 'ndvi']:
                blob[modality] = imread(filepath)
            elif modality == 'labels':
                color_labels = imread(filepath)

                def color_lookup(color):
                    """Decision tree to quickly look up class, is faster than converting
                    array into a hashable type."""
                    if color[0] == 0:
                        if color[1] == 255:
                            return 2
                        if color[1] == 60:
                            return 4
                        if color[1] == 120:
                            return 5
                        return 6
                    if color[0] == 255:
                        return 0
                    if color[0] == 170:
                        return 1
                    return 3

                labels = np.apply_along_axis(color_lookup, axis=2, arr=color_labels)
                if training_format:
                    blob['labels'] = np.array(np.array(list(range(7))) ==
                                              labels[:, :, None]).astype('int')
                else:
                    blob['labels'] = labels
            else:
                blob[modality] = one_channel_image_reader(filepath, np.uint16,
                                                          input_has_three_channels=False)
        return blob
