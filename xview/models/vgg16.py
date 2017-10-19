from tensorflow.python.layers.layers import max_pooling2d
from .custom_layers import conv2d, adap_conv

from copy import deepcopy

def vgg16(inputs, prefix, params):
    """VGG16 image encoder.

    Args:
        inputs: input tensor, of dimensions [batchsize, width, height, #channels]
        prefix: name prefix for produced weights
        params: extra parameters for convolutional layers
    Returns:
        dict of all layer names and their (intermediate) output
    """
    conv1_1 = conv2d(inputs, 64, [3, 3], name='{}_conv1_1'.format(prefix), **params)
    conv1_2 = conv2d(conv1_1, 64, [3, 3], name='{}_conv1_2'.format(prefix), **params)
    pool1 = max_pooling2d(conv1_2, [2, 2], [2, 2], name='{}_pool1'.format(prefix))
    conv2_1 = conv2d(pool1, 128, [3, 3], name='{}_conv2_1'.format(prefix), **params)
    conv2_2 = conv2d(conv2_1, 128, [3, 3], name='{}_conv2_2'.format(prefix), **params)
    pool2 = max_pooling2d(conv2_2, [2, 2], [2, 2], name='{}_pool2'.format(prefix))
    conv3_1 = conv2d(pool2, 256, [3, 3], name='{}_conv3_1'.format(prefix), **params)
    conv3_2 = conv2d(conv3_1, 256, [3, 3], name='{}_conv3_2'.format(prefix), **params)
    conv3_3 = conv2d(conv3_2, 256, [3, 3], name='{}_conv3_3'.format(prefix), **params)
    pool3 = max_pooling2d(conv3_3, [2, 2], [2, 2], name='{}_pool3'.format(prefix))
    conv4_1 = conv2d(pool3, 512, [3, 3], name='{}_conv4_1'.format(prefix), **params)
    conv4_2 = conv2d(conv4_1, 512, [3, 3], name='{}_conv4_2'.format(prefix), **params)
    conv4_3 = conv2d(conv4_2, 512, [3, 3], name='{}_conv4_3'.format(prefix), **params)
    pool4 = max_pooling2d(conv4_3, [2, 2], [2, 2], name='{}_pool4'.format(prefix))
    conv5_1 = conv2d(pool4, 512, [3, 3], name='{}_conv5_1'.format(prefix), **params)
    conv5_2 = conv2d(conv5_1, 512, [3, 3], name='{}_conv5_2'.format(prefix), **params)
    conv5_3 = conv2d(conv5_2, 512, [3, 3], name='{}_conv5_3'.format(prefix), **params)
    return {
        'conv1_1': conv1_1,
        'conv1_2': conv1_2,
        'pool1': pool1,
        'conv2_1': conv2_1,
        'conv2_2': conv2_2,
        'pool2': pool2,
        'conv3_1': conv3_1,
        'conv3_2': conv3_2,
        'conv3_3': conv3_3,
        'pool3': pool3,
        'conv4_1': conv4_1,
        'conv4_2': conv4_2,
        'conv4_3': conv4_3,
        'pool4': pool4,
        'conv5_1': conv5_1,
        'conv5_2': conv5_2,
        'conv5_3': conv5_3}


def progressive_vgg16(inputs, columns, prefix, params, adapter_params):
    """VGG16 image encoder, defined as progressive network.

    Args:
        inputs: input tensor, of dimensions [batchsize, width, height, #channels]
        columns: previously trained vgg16 encoders, given as dict:
            {<layer name>: <list of outputs from columns>}
        prefix: name prefix for produced weights
        params: extra parameters for convolutional layers
        adapter_params: parameters for adapter-blocks additional to params
    Returns:
        dict of all layer names and their (intermediate) output
    """

    all_adapter_params = deepcopy(params)
    all_adapter_params.update(adapter_params)

    # The first layer is always independent
    conv1_1 = conv2d(inputs, 64, [3, 3], name='{}_conv1_1'.format(prefix), **params)
    conv1_2 = adap_conv(conv1_1, columns['conv1_1'], 64, [3, 3],
                        name='{}_conv1_2'.format(prefix), **all_adapter_params)
    pool1 = max_pooling2d(conv1_2, [2, 2], [2, 2], name='{}_pool1'.format(prefix))
    conv2_1 = conv2d(pool1, 128, [3, 3], name='{}_conv2_1'.format(prefix), **params)
    conv2_2 = adap_conv(conv2_1, columns['conv2_1'], 128, [3, 3],
                        name='{}_conv2_2'.format(prefix), **all_adapter_params)
    pool2 = max_pooling2d(conv2_2, [2, 2], [2, 2], name='{}_pool2'.format(prefix))
    conv3_1 = conv2d(pool2, 256, [3, 3], name='{}_conv3_1'.format(prefix), **params)
    conv3_2 = conv2d(conv3_1, 256, [3, 3], name='{}_conv3_2'.format(prefix), **params)
    conv3_3 = adap_conv(conv3_2, columns['conv3_2'], 256, [3, 3],
                        name='{}_conv3_3'.format(prefix), **all_adapter_params)
    pool3 = max_pooling2d(conv3_3, [2, 2], [2, 2], name='{}_pool3'.format(prefix))
    conv4_1 = conv2d(pool3, 512, [3, 3], name='{}_conv4_1'.format(prefix), **params)
    conv4_2 = conv2d(conv4_1, 512, [3, 3], name='{}_conv4_2'.format(prefix), **params)
    conv4_3 = adap_conv(conv4_2, columns['conv4_2'], 512, [3, 3],
                        name='{}_conv4_3'.format(prefix), **all_adapter_params)
    pool4 = max_pooling2d(conv4_3, [2, 2], [2, 2], name='{}_pool4'.format(prefix))
    conv5_1 = conv2d(pool4, 512, [3, 3], name='{}_conv5_1'.format(prefix), **params)
    conv5_2 = conv2d(conv5_1, 512, [3, 3], name='{}_conv5_2'.format(prefix), **params)
    conv5_3 = adap_conv(conv5_2, columns['conv5_2'], 512, [3, 3],
                        name='{}_conv5_3'.format(prefix), **all_adapter_params)
    return {
        'conv1_1': conv1_1,
        'conv1_2': conv1_2,
        'pool1': pool1,
        'conv2_1': conv2_1,
        'conv2_2': conv2_2,
        'pool2': pool2,
        'conv3_1': conv3_1,
        'conv3_2': conv3_2,
        'conv3_3': conv3_3,
        'pool3': pool3,
        'conv4_1': conv4_1,
        'conv4_2': conv4_2,
        'conv4_3': conv4_3,
        'pool4': pool4,
        'conv5_1': conv5_1,
        'conv5_2': conv5_2,
        'conv5_3': conv5_3}
