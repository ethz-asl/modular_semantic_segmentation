import tensorflow as tf
import numpy as np
import threading
from os import path
from abc import ABCMeta, abstractmethod
from time import sleep
from types import GeneratorType

from xview.models.utils import cross_entropy
from xview.datasets.wrapper import DataWrapper


class BaseModel(object):
    """Structure for network models. Handels basic training and IO operations.

    requires the following attributes:
        prediction: usually argmax of class_probabilites, i.e. a 2D array of pixelwise
            classification
        close_queue_op: tensorflow op to close the input queue
    and requries one of the following:
        loss: a scalar value that should be minimized during training
        _train_step: a method performing one training iteration, taking a merged summary
            as input and returning the value of this summary and a loss value
    """

    __metaclass__ = ABCMeta
    required_attributes = [["loss", "_train_step"], ["prediction"], ["close_queue_op"]]

    def __init__(self, name, output_dir=None, **config):
        """Set configuration and build the model.

        Requires method _build_model to build the tensorflow graph.

        Args:
            name: The name of this model. Is used to name output files.
            output_dir: If specified, diagnostics will be saved here.
            config as key-value arguments: model-specific configurations
        """
        self.name = name
        self.output_dir = output_dir

        self.config = config

        tf.reset_default_graph()

        # Now we build the network.
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.Variable(0, trainable=False)

            self._build_graph()

            # For any child class, we require the attributes specified in the docstring
            # and defined in self.required_attributes. After self._build_graph(), we can
            # check for their existance.
            missing_attrs = ["'%s'" % attrs for attrs in self.required_attributes
                             if True not in [hasattr(self, attr) for attr in attrs]]
            if missing_attrs:
                raise AttributeError(
                    "Model class requires "
                    " and ".join(["attribute {}".format(attrs[0]) if len(attrs) < 2
                                  else " one of the attributes {}".format(attrs)
                                  for attrs in missing_attrs]))

            # To evaluate accuracy atc, we have to define soem structures here
            self.evaluation_labels = tf.placeholder(tf.float32, shape=[None, None, None])
            self.confusion_matrix = tf.confusion_matrix(
                labels=tf.reshape(self.evaluation_labels, [-1]),
                predictions=tf.reshape(self.prediction, [-1]),
                num_classes=self.config['num_classes'])

            if not hasattr(self, '_train_step'):
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                self.trainer = tf.train.AdagradOptimizer(
                    self.config['learning_rate']).minimize(
                        self.loss, global_step=self.global_step)

            self.saver = tf.train.Saver()

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            # There are local variables in the metrics
            self.sess.run(tf.local_variables_initializer())

    @abstractmethod
    def _build_graph(self):
        """Set up the whole network here."""
        raise NotImplementedError

    @abstractmethod
    def _enqueue_batch(self, batch, sess):
        """Load the given training data into the correct inputs."""
        raise NotImplementedError

    @abstractmethod
    def _evaluation_food(self, data):
        """Return a feed_dict for a prediction run on the network."""
        raise NotImplementedError

    def _load_and_enqueue(self, sess, data, coord):
        """Internal handler method for the input data queue. Will run in a seperate thread.

        Args:
            sess: The current session, needs to be the same as in the main thread.
            data: The data to load. See method predict for specifications.
            coord: The training coordinator (from tensorflow)
        """
        with self.graph.as_default():
            # We enqueue new data until it tells us to stop.
            try:
                while not coord.should_stop():
                    batch = data.next()
                    if not coord.should_stop():
                        self._enqueue_batch(batch, sess)
            except tf.errors.CancelledError:
                print('INFO: Input queue is closed, cannot enqueue any more data.')

    def fit(self, data, iterations, output=True, validation_data=None,
            validation_interval=100):
        """Train the model for given number of iterations.

        Args:
            data: A handler inheriting from DataWrapper
            iterations: The number of training iterations
            output: Boolean specifiying whether to output the loss progress
        """
        with self.graph.as_default():
            # Merge all summary creation into one op. Add summary for loss.
            loss_summary = tf.summary.scalar('loss', self.loss)
            merged_summary = tf.summary.merge_all()
            if self.output_dir is not None:
                train_writer = tf.summary.FileWriter(self.output_dir, self.graph)

            # Create a thread to load data.
            coord = tf.train.Coordinator()
            t = threading.Thread(target=self._load_and_enqueue,
                                 args=(self.sess, data, coord))
            t.start()

            # Now we can make the graph read-only.
            self.graph.finalize()

            print('INFO: Start training')
            for i in range(iterations):
                if hasattr(self, '_train_step'):
                    summary, loss = self._train_step(loss_summary)
                else:
                    summary, loss, _ = self.sess.run([loss_summary, self.loss,
                                                      self.trainer])
                if self.output_dir is not None:
                    train_writer.add_summary(summary, i)

                # Every validation_interval, we add a summary of validation values
                if i % validation_interval == 0 and validation_data is not None:
                    score, _ = self.score(validation_data)
                    summary = self.sess.run(merged_summary)
                    accuracy = tf.Summary(
                        value=[tf.Summary.Value(tag='accuracy',
                                                simple_value=score['mean_accuracy'])])

                    if output:
                        print("{:4d}: loss {:.4f}, accuracy {:.2f}".format(
                            i, loss, score['total_accuracy']))
                    if self.output_dir is not None:
                        train_writer.add_summary(accuracy, i)
                        train_writer.add_summary(summary, i)

            coord.request_stop()
            # Before we can close the queue, wait that the enqueue process stopped,
            # otherwise it will produce an error.
            sleep(20)
            self.sess.run(self.close_queue_op)
            coord.join([t])
            print('INFO: Training finished.')

    def predict(self, data):
        """Perform semantic segmentation on the input data.

        Args:
            data: a dictionary {'rgb': <array of shape [num_images, width, height]>
                                'depth': <array of shape [num_images, width, height]>}
        Returns:
            per-pixel classification of the input image in form
                <array of shape [num_images, width, height]>
        """
        with self.graph.as_default():
            return self.sess.run(self.prediction, feed_dict=self._evaluation_food(data))

    def score(self, data):
        """Measure the performance of the model with respect to the given data.

        Args:
            data: either a dictionary as for 'predict', containing keys 'rgb', 'depth'
                and 'labels', or a generator for several such dictionaries.
        Returns:
            dictionary of some measures, total confusion matrix
        """

        def get_confusion_matrix(batch):
            """Evaluate confusion matrix of network for given batch of data."""
            with self.graph.as_default():
                feed_dict = self._evaluation_food(batch)
                feed_dict[self.evaluation_labels] = batch['labels']
                confusion = self.sess.run(self.confusion_matrix, feed_dict=feed_dict)
                return np.array(confusion).astype(np.float32)

        if isinstance(data, GeneratorType):
            confusion_matrix = np.zeros((self.config['num_classes'],
                                         self.config['num_classes']))
            for batch in data:
                confusion_matrix += get_confusion_matrix(batch)
        else:
            confusion_matrix = get_confusion_matrix(data)

        # Now we compute several mesures from the confusion matrix
        measures = {}
        measures['confusion_matrix'] = confusion_matrix
        measures['accuracy'] = np.diag(confusion_matrix) / confusion_matrix.sum(1)
        measures['mean_accuracy'] = np.nanmean(measures['accuracy'])
        measures['total_accuracy'] = np.diag(confusion_matrix).sum() / \
            confusion_matrix.sum()
        measures['IU'] = np.diag(confusion_matrix) / \
            (confusion_matrix.sum(1) + confusion_matrix.sum(0) -
             np.diag(confusion_matrix))
        measures['mean_IU'] = np.nanmean(measures['IU'])
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

    def __enter__(self):
        """Contexthandler for convenience. See __exit__ for more information."""
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
            np.savez(output_path, **save_dict)
            print('INFO: Weights saved to {}'.format(output_path))
            return output_path

    def import_weights(self, filepath, translation=None, chill_mode=False,
                       warnings=True):
        """Import weights given by a numpy file. Variables are assigned to arrays which's
        key matches the variable name.

        Args:
            filepath: Full path to the file containing the weights.
            translation: Dictionary mapping variables in the network on differently named
                keys in the file.
            chill_mode: If True, ignores variables that do not match in shape and leaves
                them unassigned
        """
        with self.graph.as_default():
            weights = np.load(filepath)
            for variable in tf.global_variables():
                name = variable.op.name
                # Optimizers like Adagrad have their own variables, do not load these
                if 'grad' in name or 'Adam' in name:
                    continue
                if name not in weights and translation is not None:
                    name = translation[name]
                if name not in weights:
                    if warnings:
                        print('WARNING: {} not found in saved weights'.format(name))
                else:
                    if not variable.shape == weights[name].shape:
                        if warnings:
                            print('WARNING: wrong shape found for {}, but ignored in '
                                  'chill mode'.format(name))
                    else:
                        self.sess.run(variable.assign(weights[name]))
