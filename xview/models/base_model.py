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

            self.global_step = tf.get_variable('global_step', [1], trainable=False,
                                               initializer=tf.constant_initializer(0))

            self.saver = tf.train.Saver()

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    @abstractmethod
    def _build_graph(self):
        """Set up the whole network here."""
        raise NotImplementedError

    @abstractmethod
    def _enqueue_batch(self, batch, sess):
        """Load the given data into the occrect inputs."""
        raise NotImplementedError

    def _load_and_enqueue(self, sess, data, coord, dropout_rate):
        with self.graph.as_default():
            if not coord:
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
                    self._enqueue_batch(batch, sess)

    def _fit(self, data, iterations, output=True):
        """Train the model for given number of iterations."""

        learning_rate = self.config.get('learning_rate', 0.1)
        momentum = self.config.get('momentum', 0)

        with self.graph.as_default():
            trainer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.loss, global_step=self.global_step)

            with self.sess as sess:
                # Merge all summary creation into one op. Add summary for loss.
                tf.summary.scalar('loss', self.loss)
                merged_summary = tf.summary.merge_all()
                if self.output_dir is not None:
                    train_writer = tf.summary.FileWriter(self.output_dir, self.graph)

                # Create a thread to load data.
                coord = tf.train.Coordinator()
                t = threading.Thread(target=self._load_and_enqueue,
                                     args=(data, sess, coord,
                                           self.config['dropout_rate']))
                t.start()

                # Now we can make the graph read-only.
                self.graph.finalize()

                for i in range(iterations):
                    summary, loss, _ = sess.run([merged_summary, self.loss, trainer])
                    train_writer.add_summary(summary, i)

                    if output:
                        print("{:4d}: loss {:.4f}".format(i, loss))

                sess.run(self.close_queue_op)
                coord.request_stop()
                coord.join([t])

    def predict(self, data):
        with self.graph.as_default():
            self._load_and_enqueue(self.sess, data, False, 1.0)
            prediction = self.sess.run(self.prediction)
            self.sess.run(self.close_queue_op)
            return prediction

    def load_weights(self, path):
        """Load pretrained weights."""
        self.saver.restore(self.sess, path)

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
        return self

    def export_weights(self, save_dir=None):
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

    def import_weights(self, file):
        with self.graph.as_default():
            weights = np.load(file)
            for variable in tf.global_variables():
                name = variable.op.name
                if name not in weights:
                    print('WARNING: {} not found in saved weights'.format(name))
                else:
                    self.sess.run(variable.assign(weights[name]))
