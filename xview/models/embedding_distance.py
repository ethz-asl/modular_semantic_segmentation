import tensorflow as tf
import numpy as np
from sklearn.neighbors import BallTree, KNeighborsClassifier

from .toymodels import min_dense, min_cnn
from .simple_fcn import fcn
from .uncertainty_model import UncertaintyModel
from .utils import cross_entropy


class EmbeddingDistanceConfidence(UncertaintyModel):
    def __init__(self, prefix, data_description, modality, **config):
        self.prefix = prefix
        self.modality = modality

        standard_config = {
            'expert_model': 'fcn',
            'eps': 0.0001,
            'k': 50,
        }
        standard_config.update(config)
        UncertaintyModel.__init__(self, data_description, **standard_config)

    def _build_graph(self, train_data, test_data):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        # Training network
        with tf.name_scope('training'):

            def fast_gradient_sign(inputs, model_fn, targets, eps):
                orig_log_prob = tf.nn.log_softmax(model_fn(inputs))
                gradient = tf.gradients(cross_entropy(orig_log_prob, targets), inputs)
                # the gradient should not pass through the above computation
                adversarial_input = tf.stop_gradient(inputs + self.config['eps'] *
                                                     tf.sign(gradient))
                orig_loss = cross_entropy(orig_log_prob, targets)
                adversarial_loss = cross_entropy(
                    tf.nn.log_softmax(model_fn(adversarial_input)),
                    targets)
                return 0.5 * orig_loss + 0.5 * adversarial_loss

            train_x = train_data[self.modality]
            # ground truth labels
            train_y = tf.to_float(train_data['labels'])
            if self.config['expert_model'] == 'fcn':
                model_fn = lambda x: fcn(
                    x, self.prefix, self.config['num_units'], self.config['num_classes'],
                    is_training=True,
                    batchnorm=self.config['batch_normalization'])['score']
            elif self.config['expert_model'] == 'min_dense':
                model_fn = lambda x: min_dense(
                    x, self.config['num_units'], self.config['num_classes'])['score']
            elif self.config['expert_model'] == 'min_cnn':
                model_fn = lambda x: min_cnn(
                    x, self.config['num_units'], self.config['num_classes'],
                    is_training=True,
                    batchnorm=self.config['batch_normalization'])['score']
            else:
                raise UserWarning('expert model not found')

            self.loss = fast_gradient_sign(train_x, model_fn, train_y, self.config['eps'])

        # Network for testing / evaluation
        with tf.name_scope('testing'):
            test_x = test_data[self.modality]
            if self.config['expert_model'] == 'fcn':
                layers = fcn(test_x, self.prefix, self.config['num_units'],
                             self.config['num_classes'], is_training=False,
                             batchnorm=self.config['batch_normalization'])
                self.embedding = layers['upscore']
            elif self.config['expert_model'] == 'min_dense':
                layers = min_dense(test_x, self.config['num_units'],
                                   self.config['num_classes'], num_hidden_layers=3)
                self.embedding = layers['hidden3']
            elif self.config['expert_model'] == 'min_cnn':
                layers = min_cnn(test_x, self.config['num_units'],
                                 self.config['num_classes'], is_training=True,
                                 batchnorm=self.config['batch_normalization'])
                self.embedding = layers['2_2']
            else:
                raise UserWarning('expert model not found')
            self.prob = tf.nn.softmax(layers['score'])
            self.prediction = tf.argmax(self.prob, -1, name='label_2d')

    def fit_with_embedding(self, data, iterations, **kwargs):
        self.fit(tf.data.Dataset.from_tensor_slices(data), iterations, **kwargs)

        embeddings = self.predict(data, output_attr='embedding')
        self.knn = KNeighborsClassifier(n_neighbors=self.config['k'],
                                        algorithm='ball_tree', metric='euclidean')
        self.knn.fit(np.reshape(embeddings, [-1, embeddings.shape[-1]]),
                     data['labels'].flatten())

    def get_density(self, data):
        embeddings = self.predict(data, output_attr='embedding')
        prediction = np.expand_dims(self.predict(data), axis=-1)
        distances, idx = self.knn._tree.query(embeddings, k=self.config['k'],
                                              return_distance=True, sort_results=False)
        labels = self.knn._y[idx]
        print(embeddings.shape, distances.shape, idx.shape, labels.shape)
        return np.sum(np.exp(-distances[np.equal(labels, prediction)]), axis=-1) / \
            np.sum(np.exp(-distances), axis=-1)
