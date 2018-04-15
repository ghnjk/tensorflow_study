#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import tensorflow as tf

import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


#hyper parameter
lr = 0.001 # learning rate
trainningSteps = 2000
batchSize = 128

# 输入图像为 28*28
# 每行作为一个时间点， 总共28个时间
pixelPerTi = 28
imgTime = 28
rnnHiddenUits = 128
# 分成10类
classCount = 10

# input and output
x = tf.placeholder(tf.float32, [None, pixelPerTi * imgTime], name = "x") # 28x28
y = tf.placeholder(tf.float32, [None, classCount], name = "y")


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# rnn = Xt * Winput + biasInput -> hiddenUnit ->  hiddenUnit * Woutput + biasOutput -> Yt

def RNN(inputs, weights, biases, layerName = "RNN"):
    with tf.name_scope(layerName):
        #hiden layer for input to cell
        #input shape [None, 28, 28]
        ts = tf.reshape(inputs, [-1, pixelPerTi])
        xIn = tf.matmul(ts, weights["in"]) + biases["in"]
        xIn = tf.reshape(xIn, [-1, imgTime, rnnHiddenUits])

        #cell
        lstmCell = tf.nn.rnn_cell.BasicLSTMCell(rnnHiddenUits, forget_bias = 1.0, state_is_tuple = True)
        #lstm cell divied into two part(c_state, m_state)
        initState = lstmCell.zero_state(batchSize, dtype = tf.float32)
        output, state = tf.nn.dynamic_rnn(lstmCell, xIn, initial_state = initState, time_major = False)

        #hiden layer for output
        results = tf.matmul(state[1], weights["out"]) + biases["out"]
        ## or  把 outputs 变成 列表 [(batch, outputs)..] * steps
        # outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        # results = tf.matmul(outputs[-1], weights['out']) + biases['out']    #选取最后一个 output
    return results

weights = {
    "in": weight_variable([pixelPerTi, rnnHiddenUits])
    , "out": weight_variable([rnnHiddenUits, classCount])
}

biases = {
    "in": bias_variable([rnnHiddenUits])
    , "out": bias_variable([classCount])
}


prediction = tf.nn.softmax(RNN(x, weights, biases))

with tf.name_scope("loss"):
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction + 1e-8), reduction_indices = [1]))
    tf.summary.scalar('loss', loss) # tensorflow >= 0.12

trainStep = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.name_scope("accuracy"):
    #yPredict = sess.run(prediction, feed_dict={x: xData})
    correctPredict = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPredict, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 可视化
    logDir = "logs/rnn_mnist"
    # summaries合并
    merged = tf.summary.merge_all()
    trainWriter = tf.summary.FileWriter(logDir + '/train', sess.graph)
    # init
    sess.run(init)
    for i in range(trainningSteps):
        xBatch, yBatch = mnist.train.next_batch(batchSize)
        sess.run(trainStep, feed_dict = {
                x: xBatch
                , y: yBatch
            })
        if i % 20 == 0:
            print "train " + str(i) + " loss: " + str(
                sess.run(loss, feed_dict = {
                    x: xBatch, y: yBatch
                }))
            print sess.run(accuracy, feed_dict = {
                    x: xBatch, y: yBatch
                })
            trainResult = sess.run(merged,feed_dict = {
                    x: xBatch, y: yBatch
                })
            testResult = sess.run(merged,feed_dict = {
                    x: xBatch, y: yBatch
                })
            trainWriter.add_summary(trainResult, i)
    trainWriter.close()






