import tensorflow as tf

a = tf.constant([
        [3,3]
    ])

b = tf.constant([
        [2], [2]
    ])

product_a_b = tf.matmul(a, b)
product_b_a = tf.matmul(b, a)

with tf.Session() as sess:
    res = sess.run(product_a_b)
    print "a * b = " + str(res)
    res = sess.run(product_b_a)
    print "b * a = " + str(res)