#Installation

Requires python 2.7, python3 is not tested

    make install

Preferably call this from a virtualenvironment to avoid conflicts with system-wide python packets

Tests can be triggered with `make test`. Make sure you install the test-requirements before.

#Usage

##Models
Models are implemented following the sklearn interface, while context handling is necessary due to tensorflow semantics:

    from xview.models.simple_fcn import SimpleFCN as FCN
    from xview.data import Synthia
    data = Synthia(<config>)
    # custom configs for the model
    config = {'num_classes': 10,
              'dropout_probability': 0.2}
    with FCN(<output-directory for checkpoints and summaries>, **config) as net:
        # Train the network for 10 iterations on data.
        net.fit(data, 10)
        # Alternatively, load existing weigths.
        net.load_weights(<path to weights checkpoint>)
        net.import_weights(<path to npz file with stored weights>)
        # Now you can use it to produce classifications.
        semantic_map = net.predict({'rgb': <rgb image blob>, 'depth': <depth image blob>})

## Data
Data interfaces all expose a `obj.next()` method that returns a next batch of training data. Additionally, the SYNTHIA implementation follows the interface defined in `data_baseclass.py`, which gives access to training, test and validation sets.

# Structure
The package is devided into 3 parts: 

    - `xview/models`: implementation of classifiers
    - `xview/data`: implementation of data interfaces
    - `experiments`: scripts that implement different functionalities for experiment, following the [sacred](https://github.com/IDSIA/sacred) framework.

##Models
Any implementation of a model should inherit from `base_model.py`, which implement basic sklean interfaces such as `.fit()`, `.score()`, `.predict()` etc. aswell as training procedures and graph building.

Model implementation should be split up into a method that defines a series of tensorflow-ops mapping input to output and a class inheriting from `base_model.py` that handles all the functionality around this such as data piping etc.
In this way, models can make use of the simple network definitions without building the whole model-object.  
See `xview/models/simple_fcn.py` for an example.

Custom functions and ops used by many different models are collected in `custom_layers.py`.

##Experiments
The `experiments/utils.py` contains basic functions to interact with the sacred storage service, i.e. to load data from previous experiments and to store data from the current experiment.