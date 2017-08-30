from lib_darnn.datasets.factory import rgbd_scene
from lib_darnn.gt_single_data_layer import GtSingleDataLayer

from .wrapper import DataWrapper

available_data = {
    'rgbd_scene': [rgbd_scene, 'rgbd_scene_path']
}


class WashingtonData(DataWrapper):
    """Dataset used for DA-RNN, short RGB-D video sequences of indoor scene."""

    def __init__(self, dataset, path):
        builder, patharg = available_data[dataset]
        self.train_dataset = builder('train', **{patharg: path})

        self.train_shuffle_layer = GtSingleDataLayer(self.train_dataset.roidb,
                                                     self.train_dataset.num_classes)

    def next(self):
        batch = self.train_shuffle_layer.forward()

        # we only need one channel, they are all equal
        batch['depth'] = batch['depth'][:, :, :, 0:1]

        return batch
