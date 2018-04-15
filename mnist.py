#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import tensorflow as tf

import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, inSize, outSize, activation = None, layerName = "layer"):
    with tf.name_scope(layerName):
        Weights = tf.Variable(tf.random_normal([inSize, outSize]), name = "Weights")
        tf.summary.histogram(layerName + '/weights', Weights) # tensorflow >= 0.12
        bias = tf.Variable(tf.zeros([1, outSize]) + 0.1, name = "bias")
        tf.summary.histogram(layerName + '/bias', bias) # tensorflow >= 0.12
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + bias
        if activation != None:
            outputs = activation(Wx_plus_b)
        else:
            outputs = Wx_plus_b
        tf.summary.histogram(layerName + '/outputs', outputs) # Tensorflow >= 0.12
    return outputs


x = tf.placeholder(tf.float32, [None, 784], name = "x") # 28x28
y = tf.placeholder(tf.float32, [None, 10], name = "y")
keepProb = tf.placeholder(tf.float32, name = "keep_probability")


layer1 = add_layer(x, 784, 64, tf.nn.tanh, "layer1")
layer2 = tf.nn.dropout(layer1, keepProb, name = "layer2")
#layer3 = add_layer(layer2, 64, 32, tf.nn.tanh, "layer3")
prediction = add_layer(layer1, 64, 10, tf.nn.softmax, "prediction")

with tf.name_scope("loss"):
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction + 1e-8), reduction_indices = [1]))
    tf.summary.scalar('loss', loss) # tensorflow >= 0.12

trainStep = tf.train.AdamOptimizer().minimize(loss)


with tf.name_scope("accuracy"):
    #yPredict = sess.run(prediction, feed_dict={x: xData})
    correctPredict = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPredict, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

init = tf.global_variables_initializer()


xData = mnist.train.images
yData = mnist.train.labels
xTest = mnist.test.images
yTest = mnist.test.labels

with tf.Session() as sess:
    # 可视化
    logDir = "logs/mnist"
    # summaries合并
    merged = tf.summary.merge_all()
    trainWriter = tf.summary.FileWriter(logDir + '/train', sess.graph)
    testWriter = tf.summary.FileWriter(logDir + "/test", sess.graph)
    # init
    sess.run(init)
    for i in range(2000):
        xBatch, yBatch = mnist.train.next_batch(200)
        sess.run(trainStep, feed_dict = {
                x: xBatch
                , y: yBatch
                , keepProb: 0.6
            })
        if i % 100 == 0:
            trainResult = sess.run(merged,feed_dict = {
                    x: xData, y: yData, keepProb: 1
                })
            testResult = sess.run(merged,feed_dict = {
                    x: xTest, y: yTest, keepProb: 1
                })
            trainWriter.add_summary(trainResult, i)
            testWriter.add_summary(testResult, i)
    trainWriter.close()
    testWriter.close()



