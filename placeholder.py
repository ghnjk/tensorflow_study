#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import tensorflow as tf


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

y = tf.multiply(a, b)

with tf.Session() as sess:
    print sess.run(y, feed_dict={a: 9.0, b: 10.0})