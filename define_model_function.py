import tensorflow as tf


def weight_varibles(shape):
    with tf.name_scope("Weight"):
        initial = tf.truncated_normal(shape=shape,stddev=0.1)
        return tf.Variable(initial)

def bias_varibles(shape):
    with tf.name_scope('biases'):
        biase = tf.constant(0.1,shape=shape)
        return  tf.Variable(biase)

def conv_layer(x,W):
    with tf.name_scope('conv_layer'):
        layer = tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')
        return layer

def max_pool(x):
    with tf.name_scope('max_pool'):
        pool = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return pool