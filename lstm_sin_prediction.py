#!/usr/bin/env python
# -*- coding: UTF-8 -*- 
# @author: guohainan
# @Create: 2018-05-13
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class WebVisitPredictModel(object):

    def __init__(self, time_step=50, input_dim=1, output_dim=1):
        self.time_step = time_step
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_hidden_units = 24

        # tensorflow input
        self.x = None
        self.y = None
        # tensorflow output
        self.loss = None
        self.train_op = None
        self.y_prediction = None

        self._build_model()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.time_step, self.input_dim], name="x")
            self.y = tf.placeholder(tf.float32, shape=[None, self.output_dim], name="y")
        with tf.name_scope("lstm"):
            cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_units, state_is_tuple=True, name="lstm_cell")
            out, state = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
            out = tf.transpose(out, [1, 0, 2])
            lstm_out = tf.gather(out, int(out.get_shape()[0]) - 1, name="lstm_out")
        with tf.name_scope("dense"):
            self.y_prediction = tf.layers.dense(lstm_out, self.output_dim, name="y_prediction")
            # self.y_prediction = tf.layers.dense(self.y_prediction, self.output_dim, name="adaptor")
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.y_prediction), axis=1))
        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    ts = 50
    m = WebVisitPredictModel(time_step=ts)
    x_data = np.arange(0, 1000, 0.01)
    y_sin = np.sin(x_data) + np.cos(x_data * 2/ np.pi) + x_data / 40.0
    # for i in range(len(x_data) / ts / 5):
    #     idx = i * 5 * ts
    #     y_sin[idx: idx + ts] = y_sin[idx: idx + ts] * 2
    plt.ion()
    plt.show()
    for i in range(1000):
        x_batch = []
        y_batch = []
        for k in range(100):
            start_idx = i + 100 + k * ts
            end_idx = start_idx + ts
            x_batch.append(y_sin[start_idx: end_idx].reshape(ts, 1))
            y_batch.append([y_sin[end_idx]])
        _, loss, y_pre = m.sess.run([m.train_op, m.loss, m.y_prediction], feed_dict={
            m.x: np.array(x_batch),
            m.y: np.array(y_batch)
        })
        if i % 100 == 0:
            print("loss: ", loss)
        plt.cla()
        plt.plot(np.array(y_batch).reshape([-1, 1]))
        plt.plot(y_pre.reshape([-1, 1]))
        plt.draw()
        plt.pause(0.01)
