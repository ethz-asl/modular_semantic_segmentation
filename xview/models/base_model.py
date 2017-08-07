import tensorflow as tf
import numpy as np
import threading
from os import path
from abc import ABCMeta, abstractmethod

from xview.models.utils import cross_entropy
from xview.datasets.wrapper import DataWrapper


class BaseModel(object):
    """Structure for network models. Handels basic training and IO operations.

    requires the following attributes:
        class_probabilities: tensor output of shape [batch_size, , , num_classes]
        Y: tensor output of the ground-truth label to compare class_probabilities against
        prediction: usually argmax of class_probabilites
        close_queue_op: tensorflow op to close the input queue
    """

    __metaclass__ = ABCMeta
    required_attributes = ["class_probabilities", "Y", "prediction", "close_queue_op"]

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

        standard_config = {
            'image_width': 640,
            'image_height': 480
        }
        standard_config.update(config)
        self.config = standard_config

        # Now we build the network.
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()

            # For any child class, we require the attributes specified in the docstring
            # and defined in self.required_attributes. After self._build_graph(), we can
            # check for their existance.
            missing_attrs = ["'%s'" % attr for attr in self.required_attributes
                             if not hasattr(self, attr)]
            if missing_attrs:
                raise AttributeError("Model class requires attribute%s %s" %
                                     ("s" * (len(missing_attrs) > 1),
                                      ", ".join(missing_attrs)))

            # Loss will always be the cross-entropy.
            self.loss = tf.div(tf.reduce_sum(cross_entropy(self.Y,
                                                           self.class_probabilities)),
                               tf.reduce_sum(self.Y))

            self.global_step = tf.Variable(0, trainable=False)

            self.saver = tf.train.Saver()

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    @abstractmethod
    def _build_graph(self):
        """Set up the whole network here."""
        raise NotImplementedError

    @abstractmethod
    def _enqueue_batch(self, batch, sess):
        """Load the given data into the correct inputs."""
        raise NotImplementedError

    def _load_and_enqueue(self, sess, data, coord, dropout_rate):
        """Internal handler method for the input data queue. May run in a seperate thread.

        Args:
            sess: The current session, needs to be the same as in the main thread.
            data: The data to load. See method predict for specifications.
            coord: The training coordinator (from tensorflow)
            dropout_rate: The dropout-rate that should be enqueued with the data
        """
        with self.graph.as_default():
            if coord is None:
                # As there is no coordinator, enqueue simply one batch.
                if isinstance(data, DataWrapper):
                    # This gives one batch from the dataset.
                    batch = data.next()
                else:
                    # We otherwise assume that data is already a batch-dict.
                    batch = data
                batch['dropout_rate'] = dropout_rate
                self._enqueue_batch(batch, sess)
            else:
                # If coord is set, we enqueue new data until it tells us to stop.
                while not coord.should_stop():
                    batch = data.next()
                    batch['dropout_rate'] = dropout_rate
                    self._enqueue_batch(batch, sess)

    def fit(self, data, iterations, output=True):
        """Train the model for given number of iterations.

        Args:
            data: A handler inheriting from DataWrapper
            iterations: The number of training iterations
            output: Boolean specifiying whether to output the loss progress
        """

        learning_rate = self.config.get('learning_rate', 0.1)
        momentum = self.config.get('momentum', 0)

        with self.graph.as_default():
            trainer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.loss, global_step=self.global_step)

            # Merge all summary creation into one op. Add summary for loss.
            tf.summary.scalar('loss', self.loss)
            merged_summary = tf.summary.merge_all()
            if self.output_dir is not None:
                train_writer = tf.summary.FileWriter(self.output_dir, self.graph)

            self.sess.run(tf.global_variables_initializer())

            # Create a thread to load data.
            coord = tf.train.Coordinator()
            t = threading.Thread(target=self._load_and_enqueue,
                                 args=(self.sess, data, coord,
                                       self.config['dropout_rate']))
            t.start()

            # Now we can make the graph read-only.
            self.graph.finalize()

            for i in range(iterations):
                summary, loss, _ = self.sess.run([merged_summary, self.loss, trainer])
                train_writer.add_summary(summary, i)

                if output:
                    print("{:4d}: loss {:.4f}".format(i, loss))

            self.sess.run(self.close_queue_op)
            coord.request_stop()
            coord.join([t])

    def predict(self, data):
        """Perform semantic segmentation on the input data.

        Args:
            data: Either a handler to a dataclass inheriting from DataWrapper or a
                dictionary {'rgb': <array of shape [num_images, width, height]>
                            'depth': <array of shape [num_images, width, height]>}
        Returns:
            per-pixel classification of the input image in form
                <array of shape [num_images, width, height]>
        """
        with self.graph.as_default():
            self._load_and_enqueue(self.sess, data, None, 0.0)
            prediction = self.sess.run(self.prediction)
            self.sess.run(self.close_queue_op)
            return prediction

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

    def import_weights(self, filepath, translation=None):
        """Import weights given by a numpy file. Variables are assigned to arrays which's
        key matches the variable name.

        Args:
            filepath: Full path to the file containing the weights.
            translation: Dictionary mapping variables in the network on differently named
                keys in the file.
        """
        with self.graph.as_default():
            weights = np.load(filepath)
            for variable in tf.global_variables():
                name = variable.op.name
                if name not in weights and translation is not None:
                    name = translation[name]
                if name not in weights:
                    print('WARNING: {} not found in saved weights'.format(name))
                else:
                    self.sess.run(variable.assign(weights[name]))
