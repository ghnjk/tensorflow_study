#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import tensorflow as tf

import random
import numpy as np

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


xData = np.linspace(-1, 1, 1000)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, xData.shape)
yData = np.square(xData) - 0.5 #+ noise

xTest = np.linspace(-1, 1, 1000)[:, np.newaxis]
yTest = np.square(xTest) - 0.5

with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, 1], name = "x")
    y = tf.placeholder(tf.float32, [None, 1], name = "y")

layer1 = add_layer(x, 1, 10, tf.nn.relu, "layer1")
layer2 = add_layer(layer1, 10, 10, tf.nn.sigmoid, "layer2")

with tf.name_scope("prediction"):
    prediction = add_layer(layer2, 10, 1, activation = None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices = [1]))
    tf.summary.scalar('loss', loss) # tensorflow >= 0.12

with tf.name_scope("train"):
    trainStep = tf.train.AdamOptimizer(0.1).minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 可视化
    logDir = "logs/model_view"
    # summaries合并
    merged = tf.summary.merge_all()
    trainWriter = tf.summary.FileWriter(logDir + '/train', sess.graph)
    testWriter = tf.summary.FileWriter(logDir + '/test')
    # init
    sess.run(init)
    for epoch in range(10000):
        idx = random.sample(range(len(xData)), 100)
        xBatch = []
        yBatch = []
        for i in idx:
            xBatch.append(xData[i])
            yBatch.append(yData[i])
        sess.run(trainStep, feed_dict = {
                x: xBatch
                , y: yBatch
            })
        if epoch % 100 == 0:
            print "train " + str(epoch) + " loss: " + str(
                sess.run(loss, feed_dict = {
                    x: xData, y: yData
                }))
            rs = sess.run(merged,feed_dict = {
                    x: xData, y: yData
                })
            trainWriter.add_summary(rs)
    summary, yPredict = sess.run([merged, prediction], feed_dict = {
            x: xTest
            , y: yTest
        })
    testWriter.add_summary(summary)
    print "predict loss: " + str(
            sess.run(loss, feed_dict = {
                x: xTest
                , y: yTest
                })
        )
    trainWriter.close()
    testWriter.close()
