import tensorflow as tf
import threading
from abc import ABCMeta, abstractmethod

from xview.models.utils import cross_entropy
from xview.data.wrapper import DataWrapper


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

    def __init__(self, config, output_dir):

        standard_config = {
            'image_width': 640,
            'image_height': 480
        }
        standard_config.update(config)
        self.config = standard_config

        self.output_dir = output_dir

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
        """Load the given data into the occrect inputs."""
        raise NotImplementedError

    def _load_and_enqueue(self, sess, data, coord, keep_prob):
        with self.graph.as_default():
            if not coord:
                # As there is no coordinator, enqueue simply one batch.
                if isinstance(data, DataWrapper):
                    # This gives one batch from the dataset.
                    batch = data.next()
                else:
                    # We otherwise assume that data is already a batch-dict.
                    batch = data
                batch['keep_prob'] = keep_prob
                self._enqueue_batch(batch, sess)
            else:
                # If coord is set, we enqueue new data until it tells us to stop.
                while not coord.should_stop():
                    batch = data.next()
                    self._enqueue_batch(batch, sess)

    def fit(self, data, iterations, output=True):
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
                train_writer = tf.summary.FileWriter(self.output_dir, self.graph)

                # Create a thread to load data.
                coord = tf.train.Coordinator()
                t = threading.Thread(target=self._load_and_enqueue,
                                     args=(data, sess, coord,
                                           1 - self.config['dropout_probability']))
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

    def load(self, path):
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
