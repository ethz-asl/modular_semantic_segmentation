from xview.models.washington_fcn import FCN
from xview.data.washington_rgbd import WashingtonData
import matplotlib.pyplot as plt

data = WashingtonData('rgbd_scene',
                      '/home/hermann/Masterarbeit/code/DARNN/data/RGBDScene')

config = {'num_classes': 10,
          'dropout_probability': 0.2}

with FCN(config, '/home/hermann/Masterarbeit/experiments/test') as net:
    # net.fit(data, 10)
    net.load('/home/hermann/Masterarbeit/code/DARNN/data/RGBDScene/vgg16_fcn_rgbd_single_frame_rgbd_scene_iter_40000.ckpt')
    im = net.predict(data)

    print(im.shape)

    plt.imshow(im[0])
    plt.show()
