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
        self.keep_prob = self.net.keep_prob
        self.enqueue_op = self.net.enqueue_op
        self.close_queue_op = self.net.close_queue_op

        # the output
        self.class_probabilities = self.net.get_output('prob')

    def _load_and_enqueue(self, data, sess, coord, keep_prob):

        print('hi')
        print('data loading started')

        with self.graph.as_default():

            while not coord.should_stop():
                blobs = data.next()

                print("now I got a blob")
                print(blobs['labels'].shape)
                print(blobs["rgb"].shape)
                print(blobs['depth'].shape)

                feed_dict = {self.X_rgb: blobs['rgb'], self.X_d: blobs['depth'],
                             self.net.gt_label_2d: blobs['labels'],
                             self.keep_prob: keep_prob}
                sess.run(self.net.enqueue_op, feed_dict=feed_dict)
