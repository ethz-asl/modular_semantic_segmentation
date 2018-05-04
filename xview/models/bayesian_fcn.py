import tensorflow as tf

from .base_model import BaseModel
from .utils import cross_entropy
from .simple_fcn import encoder, decoder
from .custom_layers import entropy


def epistemic_uncertainty(inputs, pipeline, num_samples, first_run=False, **kwargs):
    """
    Samples MC-dropout samples out of the network and produces uncertainty and output

    Code adapted from Paul Sarlin

    Returns:
        Tuple, first item is class probability, second is uncertainty
    """
    counter = tf.constant(1, dtype=tf.int32, name='epistemic_sample_counter')

    # we produce a first sample to get to know the output shape and initialize the
    # tensor of samples
    samples = tf.expand_dims(pipeline(inputs, reuse=(not first_run), **kwargs), axis=0)
    # set the general shape of the tensors
    samples_shape = tf.concat([None, tf.shape(samples)[1:]])

    def loop_cond(counter, samples):
        # the loop runs until <num_samples> samples have been produced
        return tf.less(counter, num_samples)

    def loop_body(counter, samples):
        # each loop iteration adds 1 sample
        sample = pipeline(inputs, **kwargs)
        return tf.add(counter, 1), tf.concat([samples, tf.expand_dims(sample, axis=0)],
                                             axis=0)

    # now let the loop run to produce all the samples
    _, samples = tf.while_loop(loop_cond, loop_body, [counter, samples],
                               shape_invariants=[counter.get_shape(), samples_shape])
    mean = tf.reduce_mean(samples, axis=0)
    return mean, entropy(mean)


class BayesianFCN(BaseModel):
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

        BaseModel.__init__(self, 'SimpleFCN', data_description, output_dir=output_dir,
                           dropout_layers=dropout_layers, **config)

    def _build_graph(self, train_data, test_data):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        def sample_pipeline(inputs, is_training=False, reuse=True):
            # this is 1 monte-carlo sample
            encoder_layers = encoder(
                inputs, self.prefix, self.config['num_units'],
                self.config['dropout_rate'], reuse=reuse,
                dropout_layers=self.config['dropout_layers'])
            decoder_layers = decoder(
                encoder_layers['fused'], self.prefix, self.config['num_units'],
                self.config['num_classes'], is_training=True, reuse=reuse,
                dropout_rate=(self.config['dropout_rate']
                              if 'features' in self.config['dropout_layers']
                              else None))
            prob = tf.nn.softmax(decoder_layers['score'])
            return prob

        # Training input
        train_x = train_data[self.modality]
        # ground truth labels
        train_y = tf.to_float(train_data['labels'])

        # Network for testing / evaluation
        test_x = test_data[self.modality]

        if self.config['method'] == 'epistemic':
            # training pipeline
            mean, _ = epistemic_uncertainty(
                train_x, sample_pipeline, self.config['num_samples'], is_training=True,
                first_run=True)
            self.loss = tf.div(tf.reduce_sum(cross_entropy(tf.log(mean), train_y)),
                               tf.reduce_sum(train_y))

            # testing
            mean, uncertainty = epistemic_uncertainty(test_x, sample_pipeline,
                                                      self.config['num_samples'])
            self.prob = mean
            self.prediction = tf.argmax(mean, 3)
            self.uncertainty = uncertainty
