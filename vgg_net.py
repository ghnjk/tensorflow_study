#!/usr/bin/env python
# -*- coding: UTF-8 -*- 
# @author: guohainan
# @Create: 2018-05-20
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_mnist_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist


class VggNet(object):

    def __init__(self, sess, image_shape, label_count):
        self.image_shape = image_shape
        self.label_count = label_count

        # tf inputs
        self.tf_image = None
        self.tf_label = None
        # tf output
        self.tf_prediction = None
        self.tf_loss = None
        self.tf_train_op = None

        self.sess = sess
        self._build_vgg_net()
        self.sess.run(tf.global_variables_initializer())

    def _build_vgg_net(self):
        with tf.variable_scope("inputs"):
            self.tf_image = tf.placeholder(tf.float32, shape=self.image_shape, name="image")
            self.tf_label = tf.placeholder(tf.float32, shape=[None, self.label_count], name="label")

        with tf.variable_scope("VGG_NET"):
            y = self._add_vgg_block(self.tf_image, 16, 3, 2, "block_1")
            y = self._add_vgg_block(y, 32, 3, 2, "block_2")
            y = self._add_vgg_block(y, 64, 3, 3, "block_3")
            # y = self._add_vgg_block(y, 128, 3, 3, "block_4")
            # y = self._add_vgg_block(y, 128, 3, 3, "block_5")
            y = tf.layers.flatten(y, name="flattern")
            y = tf.layers.dense(y, 128, activation=tf.nn.relu, name="dense_128")
            y = tf.layers.dropout(y, rate=0.5)
            y = tf.layers.dense(y, 64, activation=tf.nn.relu, name="dense_64")
            y = tf.layers.dropout(y, rate=0.5)
            y = tf.layers.dense(y, self.label_count, activation=tf.nn.softmax, name="prediction")
            self.tf_prediction = y

        with tf.variable_scope("loss"):
            self.tf_loss = tf.reduce_mean(
                -tf.reduce_sum(
                    self.tf_label * tf.log(self.tf_prediction + 1e-8),
                    reduction_indices=1
                )
            )
            tf.summary.scalar('loss', self.tf_loss)

        with tf.variable_scope("train"):
            self.tf_train_op = tf.train.AdamOptimizer().minimize(self.tf_loss)

    @staticmethod
    def _add_vgg_block(x, filters, kernel, conv_layer_count, scope):
        with tf.variable_scope(scope):
            y = x
            for i in range(conv_layer_count):
                y = tf.layers.conv2d(
                    inputs=y,
                    filters=filters,
                    kernel_size=(kernel, kernel),
                    strides=(1, 1),
                    padding="same",
                    activation=tf.nn.relu,
                    name="conv_%d_%d_%d" % (kernel, filters, i)
                )
            y = tf.layers.max_pooling2d(
                inputs=y,
                pool_size=(2, 2),
                strides=(2, 2)
            )
        return y


def test_accuracy(mnist, net):
    x = mnist.test.images
    y = mnist.test.labels
    predict = net.sess.run(net.tf_prediction, feed_dict={
        net.tf_image: x.reshape((-1, 28, 28, 1))
    })
    y = np.argmax(y, axis=1)
    predict = np.argmax(predict, axis=1)
    success = 0
    for i in range(len(y)):
        if y[i] == predict[i]:
            success += 1
    return round(success / float(len(y)), 4)


def main():
    batch_size = 32
    max_epoch = 500
    mnist = load_mnist_data()
    loss_his = []
    accuracy_his = []
    accuracy_x_his = []
    with tf.Session() as sess:
        net = VggNet(sess=sess, image_shape=[None, 28, 28, 1], label_count=10)
        # 可视化
        log_dir = "logs/vgg_net"
        # summaries合并
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        for epoch in range(max_epoch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, loss = sess.run(
                [net.tf_train_op, net.tf_loss],
                feed_dict={
                    net.tf_image: x_batch.reshape((batch_size, 28, 28, 1)),
                    net.tf_label: y_batch.reshape(batch_size, 10)
                }
            )
            loss_his.append(loss)
            if epoch % 10 == 0:
                res = sess.run(merged, feed_dict={
                    net.tf_image: x_batch.reshape((batch_size, 28, 28, 1)),
                    net.tf_label: y_batch.reshape(batch_size, 10)
                })
                train_writer.add_summary(res, epoch)
                accuracy = test_accuracy(mnist, net)
                accuracy_his.append(accuracy)
                accuracy_x_his.append(epoch)
                print("epoch", epoch, "loss", loss, "accurace", accuracy)
    plt.subplot(211)
    plt.title("loss")
    plt.plot(loss_his)
    plt.grid()
    plt.subplot(212)
    plt.title("accuracy")
    plt.plot(accuracy_x_his, accuracy_his)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
