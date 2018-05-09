import tensorflow as tf

from .uncertainty_model import UncertaintyModel
from .utils import cross_entropy
from .simple_fcn import encoder, decoder
from .custom_layers import entropy, log_softmax


def sampling_uncertainty(inputs, pipeline, num_samples, num_classes, **kwargs):
    """
    Samples MC-dropout samples out of the network and produces uncertainty and output

    Code adapted from Paul Sarlin

    Returns:
        Tuple, first item is class probability, second is uncertainty
    """
    """
    counter = tf.constant(0, name='sample_counter')

    # we produce a first sample to get to know the output shape and initialize the
    # tensor of samples
    #samples = tf.expand_dims(pipeline(inputs, **kwargs)['prob'], axis=0)

    out_shape = tf.concat([tf.shape(inputs)[:-1],
                           tf.convert_to_tensor([num_classes])], axis=0)
    samples = tf.zeros(tf.concat([tf.convert_to_tensor([0]), out_shape], axis=0))

    # set the general shape of the tensors
    samples_shape = samples.shape
    samples_shape = tf.TensorShape(
        [None, samples_shape[1], samples_shape[2], samples_shape[3], num_classes])

    def loop_cond(counter, samples):
        # the loop runs until <num_samples> samples have been produced
        return tf.less(counter, num_samples)

    def loop_body(counter, samples):
        # each loop iteration adds 1 sample
        sample = pipeline(inputs, **kwargs)['prob']
        return tf.add(counter, 1), tf.concat([samples, tf.expand_dims(sample, axis=0)],
                                             axis=0)

    # now let the loop run to produce all the samples
    _, samples = tf.while_loop(loop_cond, loop_body, [counter, samples],
                               shape_invariants=[counter.get_shape(), samples_shape])
    """
    with tf.name_scope('samples'):
        samples = tf.stack([pipeline(inputs, **kwargs)['prob']
                            for _ in range(num_samples)], axis=0)
    mean = tf.reduce_mean(samples, axis=0)
    uncertainties = {
        'entropy': entropy(mean),
        'cond_entropy': tf.reduce_mean(entropy(samples), axis=0),
        'variance': tf.reduce_sum(tf.nn.moments(samples, axes=[0])[1], axis=-1)
    }
    return mean, uncertainties


class BayesianFCN(UncertaintyModel):
    """FCN version of BayesianFCN [http://arxiv.org/abs/1511.02680]

    Args:
        output_dir: if set, will output diagnostics and weights here
        learning_rate: learning rate of the trainer
        modality: name of the data modality, which has to be a valid key for the input
            dataset batch
        num_channels: channel-size of the input data (3 for RGB, 1 for Depth)

        Other Args see encoder and decoder
    """

    def __init__(self, prefix, data_description, modality, output_dir=None,
                 dropout_layers=['pool3', 'pool4', 'conv4_3', 'conv5_3', 'features'],
                 **config):
        self.prefix = prefix
        self.modality = modality

        UncertaintyModel.__init__(self, data_description, output_dir=output_dir,
                                  dropout_layers=dropout_layers, **config)

    def _build_graph(self, train_data, test_data):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""
        def sample_pipeline(inputs, is_training=False, reuse=tf.AUTO_REUSE):
            layers = encoder(
                inputs, self.prefix, self.config['num_units'],
                self.config['dropout_rate'], is_training=is_training, reuse=reuse,
                dropout_layers=self.config['dropout_layers'])
            layers.update(decoder(
                layers['fused'], self.prefix, self.config['num_units'],
                self.config['num_classes'], is_training=is_training, reuse=reuse,
                dropout_rate=(self.config['dropout_rate']
                              if 'features' in self.config['dropout_layers']
                              else None)))
            layers['prob'] = tf.nn.softmax(layers['score'])
            return layers

        if self.config['method'] == 'sampling':
            # training pipeline
            with tf.name_scope('training'):
                train_x = train_data[self.modality]
                # ground truth labels
                train_y = tf.to_float(train_data['labels'])

                score = sample_pipeline(train_x, is_training=True)['score']
                log_prob = log_softmax(score, self.config['num_classes'])
                self.loss = cross_entropy(log_prob, train_y)

            # testing
            with tf.name_scope('testing'):
                test_x = test_data[self.modality]
                mean, uncertainties = sampling_uncertainty(
                    test_x, sample_pipeline, self.config['num_samples'],
                    self.config['num_classes'])
                self.prediction = tf.argmax(mean, 3)
                for key, uncertainty in uncertainties.items():
                    setattr(self, key, uncertainty)
