import tensorflow as tf
import numpy as np
import time
import os
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from experiments.utils import get_mongo_observer, ExperimentData
from xview.models.fusion_fcn import fusion_fcn
from xview.models.simple_fcn import fcn
from xview.models.bayes_mix import bayes_decision_matrix
from xview.models.dirichlet_mix import dirichlet_fusion
from xview.models.variance_mix import variance_fusion


ex = Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_mongo_observer())


@ex.command
def time_fusion_fcn(net_config, repetitions):
    # cityscapes size
    rgb = tf.ones([1, 768, 384, 3])
    depth = tf.ones([1, 768, 384, 1])

    # TODO make sure this does not use batchnorm
    fused = fusion_fcn({'rgb': rgb, 'depth': depth}, {'rgb': 'rgb', 'depth': 'depth'},
                       net_config['num_units'], net_config['num_classes'],
                       trainable=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    times = []
    for _ in range(repetitions):
        start = time.time()
        result = sess.run(fused)
        end = time.time()
        times.append(end - start)

    print('Mean Time {:.2f}s, Std {:.2f}s'.format(np.mean(times), np.std(times)))


@ex.command
def time_bayes_fcn(net_config, fusion_experiment, repetitions):
    # cityscapes size
    rgb = tf.ones([1, 768, 384, 3])
    depth = tf.ones([1, 768, 384, 1])

    rgb_score = fcn(rgb, 'rgb', net_config['num_units'], net_config['num_classes'],
                    trainable=False, batchnorm=False)['score']
    depth_score = fcn(depth, 'depth', net_config['num_units'], net_config['num_classes'],
                      trainable=False, batchnorm=False)['score']

    # load confusion matrices
    record = ExperimentData(fusion_experiment).get_record()
    confusion_matrices = record['info']['confusion_matrices']
    # transform into list
    confusion_matrices = [confusion_matrices['rgb'], confusion_matrices['depth']]

    decision_matrix = tf.constant(bayes_decision_matrix(confusion_matrices))

    rgb_class = tf.argmax(tf.nn.softmax(rgb_score), 3)
    depth_class = tf.argmax(tf.nn.softmax(depth_score), 3)
    fused_class = tf.gather_nd(decision_matrix,
                               tf.stack([rgb_class, depth_class], axis=-1))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    times = []
    for _ in range(repetitions):
        start = time.time()
        result = sess.run(fused_class)
        end = time.time()
        times.append(end - start)

    print('Mean Time {:.2f}s, Std {:.2f}s'.format(np.mean(times), np.std(times)))


@ex.command
def time_dirichlet_fcn(net_config, fusion_experiment, repetitions):
    # cityscapes size
    rgb = tf.ones([1, 768, 384, 3])
    depth = tf.ones([1, 768, 384, 1])

    rgb_score = fcn(rgb, 'rgb', net_config['num_units'], net_config['num_classes'],
                    trainable=False, batchnorm=False)['score']
    depth_score = fcn(depth, 'depth', net_config['num_units'], net_config['num_classes'],
                      trainable=False, batchnorm=False)['score']
    rgb_prob = tf.nn.softmax(rgb_score, 3)
    depth_prob = tf.nn.softmax(depth_score, 3)

    # load dirichlet parameter
    record = ExperimentData(fusion_experiment).get_record()
    dirichlet_params = record['info']['dirichlet_params']
    dirichlet_config = record['config']['net_config']

    # Create all the Dirichlet distributions conditional on ground-truth class
    dirichlets = {modality: {} for modality in ['rgb', 'depth']}
    sigma = dirichlet_config['sigma']
    for c in range(net_config['num_classes']):
        for m in ('rgb', 'depth'):
            dirichlets[m][c] = tf.contrib.distributions.Dirichlet(
                sigma * dirichlet_params[m][:, c].astype('float32'),
                validate_args=False, allow_nan_stats=False)

    # Set the Prior of the classes
    data_prior = (dirichlet_params['class_counts'] /
                  (1e-20 + dirichlet_params['class_counts'].sum())).astype('float32')
    fused_score = dirichlet_fusion([rgb_prob, depth_prob], list(dirichlets.values()),
                                   data_prior)
    fused_class = tf.argmax(fused_score, 3)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    times = []
    for _ in range(repetitions):
        start = time.time()
        result = sess.run(fused_class)
        end = time.time()
        times.append(end - start)

    print('Mean Time {:.2f}s, Std {:.2f}s'.format(np.mean(times), np.std(times)))


@ex.command
def time_variance_fcn(net_config, fusion_experiment, repetitions):
    # cityscapes size
    rgb = tf.ones([1, 768, 384, 3])
    depth = tf.ones([1, 768, 384, 1])

    # load method parameter
    record = ExperimentData(fusion_experiment).get_record()
    variance_config = record['config']['net_config']

    def test_pipeline(inputs, modality):
        def sample_pipeline(inputs, modality, reuse=False):
            """One dropout sample."""
            layers = fcn(inputs, modality, net_config['num_units'],
                         net_config['num_classes'], trainable=False, is_training=False,
                         dropout_rate=variance_config['dropout_rate'],
                         dropout_layers=['pool3'], batchnorm=False)
            prob = tf.nn.softmax(layers['score'])
            return prob

        # For classification, we sample distributions with Dropout-Monte-Carlo and
        # fuse output according to variance
        samples = tf.stack([sample_pipeline(inputs, modality, reuse=(i != 0))
                            for i in range(variance_config['num_samples'])], axis=4)

        variance = tf.reduce_mean(tf.nn.moments(samples, [4])[1], axis=3,
                                  keep_dims=True)

        prob = tf.nn.softmax(fcn(inputs, modality, net_config['num_units'],
                                 net_config['num_classes'], trainable=False,
                                 is_training=False, batchnorm=False)['score'])

        # We get the label by passing the input without dropout
        return prob, variance

    rgb_prob, rgb_var = test_pipeline(rgb, 'rgb')
    depth_prob, depth_var = test_pipeline(depth, 'depth')

    fused_score = variance_fusion([rgb_prob, depth_prob], [rgb_var, depth_var])
    label = tf.argmax(fused_score, 3)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    times = []
    for _ in range(repetitions):
        start = time.time()
        result = sess.run(label)
        end = time.time()
        times.append(end - start)

    print('Mean Time {:.2f}s, Std {:.2f}s'.format(np.mean(times), np.std(times)))


@ex.command
def time_average_fcn(net_config, repetitions):
    # cityscapes size
    rgb = tf.ones([1, 768, 384, 3])
    depth = tf.ones([1, 768, 384, 1])

    rgb_score = fcn(rgb, 'rgb', net_config['num_units'], net_config['num_classes'],
                    trainable=False, batchnorm=False)['score']
    depth_score = fcn(depth, 'depth', net_config['num_units'], net_config['num_classes'],
                      trainable=False, batchnorm=False)['score']
    rgb_prob = tf.nn.softmax(rgb_score, 3)
    depth_prob = tf.nn.softmax(depth_score, 3)

    fused_class = tf.argmax(tf.reduce_mean(tf.stack([rgb_prob, depth_prob], axis=0),
                                           axis=0), 3)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    times = []
    for _ in range(repetitions):
        start = time.time()
        result = sess.run(fused_class)
        end = time.time()
        times.append(end - start)

    print('Mean Time {:.2f}s, Std {:.2f}s'.format(np.mean(times), np.std(times)))


@ex.command
def time_rgb_fcn(net_config, repetitions):
    # cityscapes size
    rgb = tf.ones([1, 768, 384, 3])

    rgb_score = fcn(rgb, 'rgb', net_config['num_units'], net_config['num_classes'],
                    trainable=False, batchnorm=False)['score']
    output_class = tf.argmax(tf.nn.softmax(rgb_score, 3), 3)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    times = []
    for _ in range(repetitions):
        start = time.time()
        result = sess.run(output_class)
        end = time.time()
        times.append(end - start)

    print('Mean Time {:.2f}s, Std {:.2f}s'.format(np.mean(times), np.std(times)))


@ex.command
def time_depth_fcn(net_config, repetitions):
    # cityscapes size
    depth = tf.ones([1, 768, 384, 1])

    depth_score = fcn(depth, 'depth', net_config['num_units'], net_config['num_classes'],
                      trainable=False, batchnorm=False)['score']
    output_class = tf.argmax(tf.nn.softmax(depth_score, 3), 3)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    times = []
    for _ in range(repetitions):
        start = time.time()
        result = sess.run(output_class)
        end = time.time()
        times.append(end - start)

    print('Mean Time {:.2f}s, Std {:.2f}s'.format(np.mean(times), np.std(times)))


if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
