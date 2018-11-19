
import tensorflow as tf
import numpy as np

x = tf.Variable(dtype=tf.float32, expected_shape=[3], initial_value=tf.zeros([3], dtype=tf.float32))
add = tf.placeholder(dtype=tf.float32, shape=x.shape)
ops = x.assign(add)

# ops = x.assign(np.random.normal(loc=0, scale=0.5, size=[3]))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        feed_dict = dict()
        np.random.seed(i)
        feed_dict[add] = np.random.normal(0, 0.5, add.shape)
        sess.run(ops, feed_dict)
        print(x.eval())

