import tensorflow as tf
import numpy as np 

with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, [None, 784])
with tf.device('/gpu:0'):
    W = tf.Variable(tf.zeros([784, 10]))
with tf.device('/gpu:0'):
    b = tf.Variable(tf.zeros([10]))
with tf.device('/cpu:0'):
    y_1 = tf.matmul(x, W)
with tf.device('/cpu:0'):
    y_2 = tf.add(y_1, b)
with tf.device('/gpu:0'):
    y = tf.nn.softmax(y_2)

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(y, feed_dict={x: np.ones((1, 784))}))