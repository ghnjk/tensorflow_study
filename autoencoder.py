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


learningRate = 0.001
imgSize = 28 * 28
layerSize = [256, 128, 64, 10]
mainFeatureSize = 2
epochs = 100
batchSize = 256
exampleToShow = 10

x = tf.placeholder(tf.float32, [None, imgSize], name = "x")

## encoder
lastSize = imgSize
lastLayer = x
with tf.name_scope("encoder"):
    for i in range(len(layerSize)): 
        layerName = "encode_%d_%d" % (i, layerSize[i])
        lastLayer = add_layer(lastLayer, lastSize, layerSize[i], tf.nn.sigmoid, layerName)
        lastSize = layerSize[i]
## 核心特征
mainFeature = add_layer(lastLayer, lastSize, mainFeatureSize, None, "mainFeature_" + str(mainFeatureSize))
lastSize = mainFeatureSize
lastLayer = mainFeature
## decoder
with tf.name_scope("decoder"):
    for i in range(len(layerSize) - 1, -1, -1):
        layerName = "decoder_%d_%d" % (len(layerSize) - i, layerSize[i])
        lastLayer = add_layer(lastLayer, lastSize, layerSize[i], tf.nn.sigmoid, layerName)
        lastSize = layerSize[i]
## prediction
prediction = add_layer(lastLayer, lastSize, imgSize, tf.nn.sigmoid, "prediction")

## loss
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.pow(prediction - x, 2))
    tf.summary.scalar('loss', loss)

## train
train = tf.train.AdamOptimizer(learningRate).minimize(loss)


def show_prediction_img(sess):
    xTrue = mnist.test.images
    xPred = sess.run(prediction, feed_dict = {x: xTrue})
    f, a = plt.subplots(2, exampleToShow, figsize = (10, 2))
    for i in range(exampleToShow):
        a[0][i].imshow(np.reshape(xData[i], (28, 28)))
        a[1][i].imshow(np.reshape(xPred[i], (28, 28)))
    plt.show()

def show_image_main_feature(sess):
    xTrue = mnist.test.images
    mfRes = sess.run(mainFeature, feed_dict = {x: xTrue})
    plt.scatter(mfRes[:, 0], mfRes[:, 1], c = mnist.test.labels)
    plt.show()

with tf.Session() as sess:
    # 可视化
    logDir = "logs/auto_encoder"
    # summaries合并
    merged = tf.summary.merge_all()
    trainWriter = tf.summary.FileWriter(logDir + '/train', sess.graph)
    # init
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        batchCount = int(mnist.train.num_examples/batchSize)
        for j in range(batchCount):
            xData, _ = mnist.train.next_batch(batchSize)
            sess.run([train, merged], feed_dict = {x: xData})
        res, c = sess.run([merged, loss], feed_dict = {x:mnist.train.images})
        trainWriter.add_summary(res, i)
        print "epose %d : loss: %0.4lf" % (i, c)
    print "train finished."
    trainWriter.close()
    show_prediction_img(sess)
    show_image_main_feature(sess)




