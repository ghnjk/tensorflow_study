#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import tensorflow as tf

count = tf.Variable(0, name = "count")

one = tf.constant(1)

x = count
for i in range(9):
    x = tf.add(x, one)

updateCount = tf.assign(count, x)

# 如果定义 Variable, 就一定要 initialize
init = tf.global_variables_initializer()  # 替换成这样就好

with tf.Session() as sess:
    res = sess.run(init)
    print "init: " + str(res)
    for i in range(3):
        sess.run(updateCount)
        print "count: " + str(sess.run(count))

