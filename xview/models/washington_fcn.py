from lib_darnn.networks.vgg16_convs import vgg16_convs
from .base_model import BaseModel


class FCN(BaseModel):

    def _build_graph(self):
        # set up the washington code
        self.net = vgg16_convs('RGBD', self.config['num_classes'], 64, [1])

        # the input
        self.X_rgb = self.net.data  # rgb channel
        self.X_d = self.net.data_p  # depth channel
        self.Y = self.net.get_output("gt_label_2d")  # ground truth label
        self.prediction = self.net.get_output("label_2d")
        self.keep_prob = self.net.keep_prob
        self.enqueue_op = self.net.enqueue_op
        self.close_queue_op = self.net.close_queue_op

        # the output
        self.class_probabilities = self.net.get_output('prob')

    def _enqueue_batch(self, batch, sess):
        with self.graph.as_default():

            print(batch['labels'].shape)
            print(batch["rgb"].shape)
            print(batch['depth'].shape)

            feed_dict = {self.X_rgb: batch['rgb'], self.X_d: batch['depth'],
                         self.net.gt_label_2d: batch['labels'],
                         self.keep_prob: batch['keep_prob']}
            sess.run(self.net.enqueue_op, feed_dict=feed_dict)
