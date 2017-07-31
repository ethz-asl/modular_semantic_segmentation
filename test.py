from xview.models.washington_fcn import FCN
from xview.data.washington_rgbd import WashingtonData


data = WashingtonData('rgbd_scene', '/home/hermann/Masterarbeit/code/DARNN/data/RGBDScene')

config = {'num_classes': 10,
          'dropout_probability': 0.2}

net = FCN(config, '/home/hermann/Masterarbeit/experiments/test')
net.fit(data, 10)
