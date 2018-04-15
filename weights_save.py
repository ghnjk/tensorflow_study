#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import tensorflow as tf




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, dtype = tf.float32, name = "Weight")

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, dtype = tf.float32, name = "bias")

# remember to define the same dtype and shape when restore
W = weight_variable([2,2,4,2])
b = bias_variable([2])


init = tf.global_variables_initializer()

saver = tf.train.Saver()

def save_weight():
    with tf.Session() as sess:
        # 可视化
        logDir = "logs"
        weightPath = logDir + "/weight.ckpt"
        sess.run(init)
        print "savve at : " + str(saver.save(sess, weightPath))

def load_weight():
    with tf.Session() as sess:
        logDir = "logs"
        weightPath = logDir + "/weight.ckpt"
        saver.restore(sess, weightPath)
        print("weights: ", sess.run(W))
        print("bias: ", sess.run(b))

if __name__ == '__main__':
    load_weight()
