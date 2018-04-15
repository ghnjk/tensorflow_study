#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import tensorflow as tf

import random
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, inSize, outSize, activation = None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    bias = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + bias
    if activation != None:
        outputs = activation(Wx_plus_b)
    else:
        outputs = Wx_plus_b
    return outputs


xData = np.linspace(-1, 1, 1000)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, xData.shape)
yData = np.square(xData) - 0.5 #+ noise

xTest = np.linspace(-1, 1, 1000)[:, np.newaxis]
yTest = np.square(xTest) - 0.5

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

layer1 = add_layer(x, 1, 10, tf.nn.relu)
layer2 = add_layer(layer1, 10, 10, tf.nn.sigmoid)

prediction = add_layer(layer2, 10, 1, activation = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices = [1]))

trainStep = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
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
    yPredict = sess.run(prediction, feed_dict = {
            x: xTest
        }).tolist()
    print "predict loss: " + str(
            sess.run(loss, feed_dict = {
                x: xTest
                , y: yTest
                })
        )

plt.figure()

plt.plot(xTest, yTest, label = "real")
plt.plot(xTest, yPredict, label = "predict")
plt.legend()
plt.title("predict for y = x ^ 2 - 0.5")
plt.show()
