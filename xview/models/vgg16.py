from tensorflow.python.layers.layers import max_pooling2d, dropout
from .custom_layers import conv2d


def vgg16(inputs, prefix, params):
    """VGG16 image encoder.

    Args:
        inputs: input tensor, of dimensions [batchsize, width, height, #channels]
        prefix: name prefix for produced weights
        params: extra parameters for convolutional layers
    Returns:
        output of last convolutional layers after 3rd and 4th pooling
            (conv4_3 and conv5_3)
    """
    net = conv2d(inputs, 64, [3, 3], name='{}_conv1_1'.format(prefix), **params)
    net = conv2d(net, 64, [3, 3], name='{}_conv1_2'.format(prefix), **params)
    net = max_pooling2d(net, [2, 2], [2, 2], name='{}_pool1'.format(prefix))
    net = conv2d(net, 128, [3, 3], name='{}_conv2_1'.format(prefix), **params)
    net = conv2d(net, 128, [3, 3], name='{}_conv2_2'.format(prefix), **params)
    net = max_pooling2d(net, [2, 2], [2, 2], name='{}_pool2'.format(prefix))
    net = conv2d(net, 256, [3, 3], name='{}_conv3_1'.format(prefix), **params)
    net = conv2d(net, 256, [3, 3], name='{}_conv3_2'.format(prefix), **params)
    net = conv2d(net, 256, [3, 3], name='{}_conv3_3'.format(prefix), **params)
    net = max_pooling2d(net, [2, 2], [2, 2], name='{}_pool3'.format(prefix))
    net = conv2d(net, 512, [3, 3], name='{}_conv4_1'.format(prefix), **params)
    net = conv2d(net, 512, [3, 3], name='{}_conv4_2'.format(prefix), **params)
    conv4 = conv2d(net, 512, [3, 3], name='{}_conv4_3'.format(prefix), **params)
    net = max_pooling2d(conv4, [2, 2], [2, 2], name='{}_pool4'.format(prefix))
    net = conv2d(net, 512, [3, 3], name='{}_conv5_1'.format(prefix), **params)
    net = conv2d(net, 512, [3, 3], name='{}_conv5_2'.format(prefix), **params)
    conv5 = conv2d(net, 512, [3, 3], name='{}_conv5_3'.format(prefix), **params)
    return conv4, conv5
