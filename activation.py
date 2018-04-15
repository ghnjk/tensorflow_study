#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32, [20])
yVal = tf.constant(range(-10, 10), dtype = tf.float32)

afList = []
afList.append({
    "name": "relu"
    , "af": tf.nn.relu(x, name = "relu")
    })
afList.append({
    "name": "dropout"
    , "af": tf.nn.dropout(x, keep_prob = 0.2, name = "dropout")
    })
afList.append({
    "name": "tanh"
    , "af": tf.nn.tanh(x, name = "tanh")
    })
afList.append({
    "name": "sigmoid"
    , "af": tf.nn.sigmoid(x, name = "sigmoid")
    })
afList.append({
    "name": "softmax"
    , "af": tf.nn.softmax(x, name = "softmax")
    })

yList = []

with tf.Session() as sess:
    for i in range(len(afList)):
        y = sess.run(afList[i]["af"], feed_dict = {x: range(-10, 10)})
        yList.append(y.tolist())

plt.figure()

for i in range(len(yList)):
    plt.subplot((len(yList) + 1)/2 * 100 + 20 + i)
    plt.plot(range(-10,10), yList[i], label = afList[i]["name"])
    #plt.title(afList[i]["name"])
    plt.legend() # 显示图例

plt.show()