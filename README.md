#Installation

Requires python 2.7, python3 is not tested

    cd .../xview/semantic-segmentation
    pip install -r requirements.txt
    pip install .
    sh download_data.sh <path to data base-directory>

If you want to use the package while developing, consider installing it (instead than the 3rd line above) like this:
    
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
        
        # Train the network for 10 iterations on data.
        net.fit(data, 10)
        # Alternatively, load existing weigths.
        net.load(<path to weights checkpoint>)
        # Now you can use it to produce classifications.
        semantic_map = net.predict({'rgb': <rgb image blob>, 'depth': <depth image blob>})

