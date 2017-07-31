import tensorflow as tf
import threading
import time

from utils import define_scope, cross_entropy


class BaseModel(object):

    def __init__(self, config, output_dir):

        standard_config = {
            'image_width': 640,
            'image_height': 480
        }
        standard_config.update(config)
        self.config = standard_config

        self.output_dir = output_dir

        # build the network
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()

            self.loss = tf.div(tf.reduce_sum(cross_entropy(self.Y, self.class_probabilities)),
                               tf.reduce_sum(self.Y))

            self.global_step = tf.Variable(0, trainable=False)

    def _load_and_enqueue(self, data, sess, coord, keep_prob):
        """Load the given data into the occrect inputs."""

        raise NotImplementedError
    
    def fit(self, data, iterations, output=True):
        """Train the model for given number of iterations."""

        learning_rate = self.config.get('learning_rate', 0.1)
        momentum = self.config.get('momentum', 0)

        with self.graph.as_default():
            trainer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)  # global_step=self.global_step

            with tf.Session() as sess:
                # add summaries
                tf.summary.scalar('loss', self.loss)
                merged_summary = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(self.output_dir, self.graph)

                # create a thread to load data
                coord = tf.train.Coordinator()
                t = threading.Thread(target=self._load_and_enqueue,
                                     args=(data, sess, coord, 1 - self.config['dropout_probability']))
                t.start()

                # initialize variables
                sess.run(tf.global_variables_initializer())
                self.graph.finalize()

                for i in range(iterations):
                    summary, loss, _ = sess.run([merged_summary, self.loss, trainer])
                    train_writer.add_summary(summary, i)

                    if output:
                        print("{:4d}: loss {:.4f}".format(i, loss))

                sess.run(self.close_queue_op)
                coord.request_stop()
                coord.join([t])