#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import tensorflow as tf

import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def add_conv2d(x, W):
    #strides 每隔多少取样. stride = [1, x_movement, y_movement, 1] 第一个和第四个必须为1
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = "SAME")#

def add_max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

#
x = tf.placeholder(tf.float32, [None, 784], name = "x") # 28x28
y = tf.placeholder(tf.float32, [None, 10], name = "y")
keepProb = tf.placeholder(tf.float32, name = "keep_probability")

#conv2d layer1
with tf.name_scope("conv2d_layer1"):
    xImage = tf.reshape(x, [-1, 28, 28, 1])
    Wlayer1 = weight_variable([5, 5, 1, 32]) #input patchSize/kernel = [5, 5],  --> output [1, 32] input depth: 1  , depth 32
    biasLayer1 = bias_variable([32]) # must equals to W's depth
    layer1 = tf.nn.relu(add_conv2d(xImage, Wlayer1) + biasLayer1) # output : [28, 28, 32]
    layer1Pool = add_max_pool_2x2(layer1)# output : [14, 14, 32]
#conv2d layer2
with tf.name_scope("conv2d_layer2"):
    Wlayer2 = weight_variable([5, 5, 32, 64]) #input patchSize/kernel = [5, 5],  --> output [32, 64] 32 point , depth 64
    biasLayer2 = bias_variable([64]) # must equals to W's depth
    layer2 = tf.nn.relu(add_conv2d(layer1Pool, Wlayer2) + biasLayer2) # output : [14, 14, 64]
    layer2Pool = add_max_pool_2x2(layer2)# output : [7, 7, 32]
#fullconnection_layer3
with tf.name_scope("fullconnection_layer3"):
    Wlayer3 = weight_variable([7 * 7 * 64, 1024])
    biasLayer3 = bias_variable([1024])
    flatPool2 = tf.reshape(layer2Pool, [-1, 7 * 7 * 64])
    layer3 = tf.nn.relu(tf.matmul(flatPool2, Wlayer3) + biasLayer3)
    layer3Dropout = tf.nn.dropout(layer3, keepProb)
#fullconnection_layer4
with tf.name_scope("fullconnection_layer4"):
    Wlayer4 = weight_variable([1024, 10])
    biasLayer4 = bias_variable([10])
    layer4 = tf.nn.softmax(tf.matmul(layer3Dropout, Wlayer4) + biasLayer4)

prediction = layer4

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
    logDir = "logs/cnn_mnist"
    # summaries合并
    merged = tf.summary.merge_all()
    trainWriter = tf.summary.FileWriter(logDir + '/train', sess.graph)
    testWriter = tf.summary.FileWriter(logDir + "/test", sess.graph)
    # init
    sess.run(init)
    for i in range(100):
        xBatch, yBatch = mnist.train.next_batch(32)
        sess.run(trainStep, feed_dict = {
                x: xBatch
                , y: yBatch
                , keepProb: 0.6
            })
        if i % 2 == 0:
            print "train " + str(i) + " loss: " + str(
                sess.run(loss, feed_dict = {
                    x: xData, y: yData, keepProb: 1
                }))
            print sess.run(accuracy, feed_dict = {
                    x: xData, y: yData, keepProb: 1
                })
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



