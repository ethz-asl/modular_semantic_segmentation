import tensorflow as tf
import numpy as np
from os import path
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from tqdm import tqdm


class BaseModel(object):
    """Structure for network models. Handels basic training and IO operations.

    requires the following attributes:
        prediction: usually argmax of class_probabilites, i.e. a 2D array of pixelwise
            classification
        close_queue_op: tensorflow op to close the input queue
        loss: a scalar value that should be minimized during training
    if initialized with supports_training=False:
        only required attribute is self.precition
    """

    __metaclass__ = ABCMeta
    required_attributes = [["loss"], ["prediction"]]

    def __init__(self, name, data_description, output_dir=None, supports_training=True,
                 batchsize=1, **config):
        """Set configuration and build the model.

        Requires method _build_model to build the tensorflow graph.

        Args:
            name: The name of this model. Is used to name output files.
            output_dir: If specified, diagnostics will be saved here.
            config as key-value arguments: model-specific configurations
        """
        self.name = name
        self.output_dir = output_dir
        self.supports_training = supports_training
        self.config = config
        self.config['batchsize'] = batchsize
        self.config['num_classes'] = data_description[2]
        # add a new dim for the batchsize into the data description
        self.testdata_description = [data_description[0], {
            key: [None, *description]
            for key, description in data_description[1].items()}]
        # data does have a one-hot format in the labels
        data_description = deepcopy(self.testdata_description[1])
        data_description['labels'] = list(data_description['labels'])
        data_description['labels'].append(self.config['num_classes'])
        self.data_description = [self.testdata_description[0], data_description]

        self._initialize_graph()

    def _initialize_graph(self):
        # Now we build the network.
        self.graph = tf.Graph()
        self.graph_context = self.graph.as_default()
        with self.graph.as_default():
            # inference data coming packed as tf.data.dataset, therefore we create
            # a feedable dataset iterator
            self.training_handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                self.training_handle, *self.data_description)
            training_batch = iterator.get_next()

            self.testing_handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                self.testing_handle, *self.testdata_description)
            test_batch = iterator.get_next()
            evaluation_labels = test_batch['labels']

            self._build_graph(training_batch, test_batch)

            # For any child class, we require the attributes specified in the docstring
            # and defined in self.required_attributes. After self._build_graph(), we can
            # check for their existance.
            if self.supports_training:
                missing_attrs = ["'%s'" % attrs for attrs in self.required_attributes
                                 if True not in [hasattr(self, attr) for attr in attrs]]
                if missing_attrs:
                    raise AttributeError(
                        "Model class requires "
                        " and ".join(["attribute {}".format(attrs[0]) if len(attrs) < 2
                                      else " one of the attributes {}".format(attrs)
                                      for attrs in missing_attrs]))
            else:
                if not hasattr(self, 'prediction'):
                    raise AttributeError("Model class required attribute prediction")

            # To evaluate accuracy atc, we have to define some structures here
            self.confusion_matrix = tf.confusion_matrix(
                labels=tf.reshape(evaluation_labels, [-1]),
                predictions=tf.reshape(self.prediction, [-1]),
                num_classes=self.config['num_classes'])

            if self.supports_training:
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    trainers = {'adagrad': tf.train.AdagradOptimizer,
                                'adam': tf.train.AdamOptimizer,
                                'rmsprop': tf.train.RMSPropOptimizer}
                    self.trainer = trainers[self.config.get('trainer', 'adam')](
                        learning_rate=self.config.get('learning_rate', 0.0001)).minimize(
                            self.loss, global_step=self.global_step)

            self.saver = tf.train.Saver()

            # Limit GPU fraction
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=self.config.get('gpu_fraction', 1))
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.sess.run(tf.global_variables_initializer())
            # There are local variables in the metrics
            self.sess.run(tf.local_variables_initializer())

    @abstractmethod
    def _build_graph(self):
        """Set up the whole network here."""
        raise NotImplementedError

    def fit(self, dataset, iterations, output=True, validation_dataset=None,
            validation_interval=100, additional_eval_datasets={}):
        """Train the model for given number of iterations.

        Args:
            dataset: tf.data.Dataset
            iterations: The number of training iterations
        """
        if not self.supports_training:
            raise UserWarning("ERROR: Model %s does not support training" % self.name)

        with self.graph.as_default():
            # Merge all summary creation into one op. Add summary for loss.
            tf.summary.scalar('loss', self.loss)
            merged_summary = tf.summary.merge_all()
            if self.output_dir is not None:
                train_writer = tf.summary.FileWriter(self.output_dir)

            # initialize datasets
            def _onehot_mapper(blob):
                blob['labels'] = tf.one_hot(blob['labels'], self.config['num_classes'],
                                            dtype=tf.int32)
                return blob

            train_iterator = dataset.map(_onehot_mapper, 10)\
                .batch(self.config['batchsize'])\
                .make_one_shot_iterator()
            train_handle = self.sess.run(train_iterator.string_handle())

            if validation_dataset is None:
                validation_iterator = dataset.take(10).batch(self.config['batchsize'])\
                    .make_initializable_iterator()
            else:
                validation_iterator = validation_dataset.batch(self.config['batchsize'])\
                    .make_initializable_iterator()

            print('INFO: Start training')
            for i in tqdm(range(iterations), disable=output):
                # Every validation_interval, we run the summary and validation values
                if i % validation_interval == 0 and validation_dataset is not None:
                    _, summary = self.sess.run(
                        (self.trainer, merged_summary),
                        feed_dict={self.training_handle: train_handle})
                    score, _ = self.score(validation_iterator)
                    accuracy = tf.Summary(
                        value=[tf.Summary.Value(tag='accuracy',
                                                simple_value=score['total_accuracy'])])
                    iou = tf.Summary(
                        value=[tf.Summary.Value(tag='IoU',
                                                simple_value=score['mean_IoU'])])

                    if output:
                        print("{:4d}: accuracy {:.2f}, IoU {:.2f}".format(
                            i, score['total_accuracy'], score['mean_IoU']))
                    if self.output_dir is not None:
                        train_writer.add_summary(accuracy, i)
                        train_writer.add_summary(iou, i)
                        train_writer.add_summary(summary, i)

                    # Add additional summaries if specified
                    for key, additional_dataset in additional_eval_datasets.items():
                        val = self.score(additional_dataset)[0]['mean_IoU']
                        summary = tf.Summary(value=[tf.Summary.Value(tag=key,
                                                                     simple_value=val)])
                        train_writer.add_summary(summary, i)

                    if 'abort_at_iou' in self.config:
                        if score['mean_IoU'] > self.config['abort_at_iou']:
                            break
                else:
                    self.sess.run(self.trainer,
                                  feed_dict={self.training_handle: train_handle})
            print('INFO: Training finished.')

    def predict(self, data, output_prob=False):
        """Perform semantic segmentation on the input data.

        Args:
            data: a dictionary {'rgb': <array of shape [num_images, width, height]>
                                'depth': <array of shape [num_images, width, height]>}
            output_prob: a boolean indicting if classe probabilities should be outputted
                instead of the label of the most likely class
        Returns:
            per-pixel classification of the input image in form:
                - <array of shape [num_images, width, height, num_classes]>
                  if output_prob is specified and true
                - <array of shape [num_images, width, height]> else
        """
        with self.graph.as_default():
            # initialize the data dependent on the type
            if isinstance(data, tf.data.Dataset):
                iterator = data.batch(self.config['batchsize']).make_one_shot_iterator()
            else:
                iterator = tf.data.Dataset.from_tensor_slices(data)\
                    .batch(self.config['batchsize']).make_one_shot_iterator()
            handle = self.sess.run(iterator.string_handle())
            output = self.prob if (output_prob and self.prob) else self.prediction

            # collect all the batches in this list
            ret = []
            while True:
                try:
                    ret.append(self.sess.run(output,
                                             feed_dict={self.testing_handle: handle}))
                except tf.errors.OutOfRangeError:
                    break
            return np.concatenate(ret)

    def score(self, data, max_iterations=None):
        """Measure the performance of the model with respect to the given data.

        Args:
            data: either a dictionary as for 'predict', containing keys 'rgb', 'depth'
                and 'labels', or a generator for several such dictionaries.
        Returns:
            dictionary of some measures, total confusion matrix
        """
        with self.graph.as_default():
            # initialize the data dependent on the type
            if isinstance(data, tf.data.Dataset):
                iterator = data.batch(self.config['batchsize']).make_one_shot_iterator()
            elif isinstance(data, tf.data.Iterator):
                iterator = data
                self.sess.run(iterator.initializer)
            else:
                iterator = tf.data.Dataset.from_tensor_slices(data)\
                    .batch(self.config['batchsize']).make_one_shot_iterator()
            handle = self.sess.run(iterator.string_handle())

            # run through batches of the data and collect all in this confusion matrix
            confusion_matrix = np.zeros((self.config['num_classes'],
                                         self.config['num_classes']))
            while True:
                try:
                    confusion_matrix += self.sess.run(
                        self.confusion_matrix, feed_dict={self.testing_handle: handle})
                except tf.errors.OutOfRangeError:
                    break

        with np.errstate(divide='ignore', invalid='ignore'):
            # Now we compute several mesures from the confusion matrix
            measures = {}
            measures['confusion_matrix'] = confusion_matrix
            measures['recall'] = np.diag(confusion_matrix) / confusion_matrix.sum(1)
            measures['precision'] = np.diag(confusion_matrix) / confusion_matrix.sum(0)
            measures['F1'] = 2 * measures['precision'] * measures['recall'] / \
                (measures['precision'] + measures['recall'])
            measures['mean_F1'] = np.nanmean(measures['F1'])
            measures['total_accuracy'] = np.diag(confusion_matrix)[1:].sum() / \
                confusion_matrix[1:, :].sum()
            measures['IoU'] = np.diag(confusion_matrix) / \
                (confusion_matrix.sum(1) + confusion_matrix.sum(0) -
                 np.diag(confusion_matrix))
            measures['mean_IoU'] = np.nanmean(measures['IoU'][1:])

        return measures, confusion_matrix

    def load_weights(self, filepath):
        """Load model weights stored in a tensorflow checkpoint.

        Args:
            filepath: Full path to the ckeckpoint
        """
        self.saver.restore(self.sess, filepath)

    def close(self):
        """Close session of this model."""
        self.sess.close()

    def __exit__(self, *args):
        """Contexthandler for convenience.

        Use like this
            With Model() as net:
                net.fit()
                net.predict()
        """
        self.close()
        self.graph_context.__exit__(*args)

    def __enter__(self):
        """Contexthandler for convenience. See __exit__ for more information."""
        self.graph_context.__enter__()
        return self

    def export_weights(self, save_dir=None):
        """Export weights as numpy arrays and write them into a file. The key to each
        array is the name of the variable.

        Args:
            save_dir: Directory the weights are saved to. Only need to specify if
                output_dir was not set at model instantiation. Overwrites output_dir.
        Returns:
            Full path to the output file.
        """
        # Check if there is a location to write to.
        if save_dir is None and self.output_dir is None:
            print('ERROR: No path specified to save weights to.')
            return

        with self.graph.as_default():
            # We will create a dict with all variable values as numpy arrays and then
            # save it.
            save_dict = {}
            for variable in tf.global_variables():
                value = self.sess.run(variable)
                save_dict[variable.op.name] = value

            output_path = save_dir
            if output_path is None:
                output_path = self.output_dir
            # Add the filename indicating model name and step to the path.
            step = int(self.sess.run(self.global_step))
            output_path = path.join(output_path, '{}_weights_{}.npz'.format(self.name,
                                                                            step))
            np.savez_compressed(output_path, **save_dict)
            print('INFO: Weights saved to {}'.format(output_path))
            return output_path

    def import_weights(self, filepath, translate_prefix=False, chill_mode=False,
                       warnings=True):
        """Import weights given by a numpy file. Variables are assigned to arrays which's
        key matches the variable name.

        Args:
            filepath: Full path to the file containing the weights.
            translate_prefix: If set, will translate weights with a different prefix
                into the given prefix
            chill_mode: If True, ignores variables that do not match in shape and leaves
                them unassigned
        """
        with self.graph.as_default():
            initializers = []
            weights = np.load(filepath, mmap_mode='w+')
            import_prefix = weights.keys()[0].split('/')[0].split('_')[0]

            def translate_name(name):
                """May translate the prefix of the name according to settings."""
                if not translate_prefix:
                    return name
                if not name.startswith(translate_prefix):
                    return name
                splitted = name.split('/')
                further_splitted = splitted[0].split('_')
                # dirty fix for forest
                if further_splitted[0] == 'forest':
                    return name
                # exchange prefix
                further_splitted[0] = import_prefix
                splitted[0] = '_'.join(further_splitted)
                return '/'.join(splitted)

            for variable in tf.global_variables():
                name = translate_name(variable.op.name)
                # Optimizers like Adagrad have their own variables, do not load these
                if 'grad' in name or 'Adam' in name or 'RMS' in name:
                    continue
                if name not in weights:
                    if warnings:
                        print('WARNING: {} not found in saved weights'.format(name))
                else:
                    if not variable.shape == weights[name].shape:
                        if warnings:
                            print('WARNING: wrong shape found for {}, but ignored in '
                                  'chill mode'.format(name))
                            print('stored shape: ', weights[name].shape,
                                  'expected shape: ', variable.shape)
                        if chill_mode:
                            initializers.append(variable.assign(weights[name]))
                    else:
                        initializers.append(variable.assign(weights[name]))
            self.sess.run(initializers)
