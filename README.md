#Installation

Requires python 2.7, python3 is not tested

    cd .../xview/semantic-segmentation
    pip install -r requirements
    pip install -e .

#Usage

##Models
Models are implemented following the sklearn interface, while context handling is necessary due to tensorflow semantics:

    from xview.models.washington_fcn import FCN
    from xview.data.washington_rgbd import WashingtonData
    data = WashingtonData('rgbd_scene', <path_to_data>)
    # custom configs for the model
    config = {'num_classes': 10,
              'dropout_probability': 0.2}
    with FCN(config, <output-directory for checkpoints and summaries>) as net:
        net.fit(data, 10)
        semantic_map = net.predict({'rgb': <rgb image blob>, 'depth': <depth image blob>})

