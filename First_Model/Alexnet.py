### Define your architecture here.

import tensorflow as tf
from tensorflow.contrib.layers import flatten
import time
from sklearn.utils import shuffle

def Alexnet(x):    
    # Layer 1
    F_W_1 = tf.get_variable('W1',
                               shape=[11, 11, 3, 96],
                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                          mode='FAN_IN',
                                                                                          uniform=False,
                                                                                          seed=None,
                                                                                          dtype=tf.float32))
    F_b_1 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[96]))    
    conv = tf.nn.conv2d(x, F_W_1, [1, 4, 4, 1], padding='SAME')
    layer_conv1 = tf.nn.bias_add(conv, F_b_1)
    layer_activation1 = tf.nn.relu(layer_conv1)
    mean, var = tf.nn.moments(layer_activation1, [0, 1, 2])
    batch_norm1 = tf.nn.batch_normalization(layer_activation1, mean, var, 0, 1, 0)
    layer_pooling1 = tf.nn.max_pool(batch_norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer 2

    F_W_2 = tf.get_variable('W2',
                               shape=[5, 5, 96, 256],
                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                          mode='FAN_IN',
                                                                                          uniform=False,
                                                                                          seed=None,
                                                                                          dtype=tf.float32))
    F_b_2 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256]))    
    conv = tf.nn.conv2d(layer_pooling1, F_W_2, [1, 1, 1, 1], padding='SAME')
    layer_conv2 = tf.nn.bias_add(conv, F_b_2)
    layer_activation2 = tf.nn.relu(layer_conv2)
    mean, var = tf.nn.moments(layer_activation2, [0, 1, 2])
    batch_norm2 = tf.nn.batch_normalization(layer_activation2, mean, var, 0, 1, 0)
    layer_pooling2 = tf.nn.max_pool(batch_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # Layer 3
    F_W_3 = tf.get_variable('W3',
                               shape=[3, 3, 256, 384],
                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                          mode='FAN_IN',
                                                                                          uniform=False,
                                                                                          seed=None,
                                                                                          dtype=tf.float32))
    F_b_3 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384]))    
    conv = tf.nn.conv2d(layer_pooling2, F_W_3, [1, 1, 1, 1], padding='SAME')
    layer_conv3 = tf.nn.bias_add(conv, F_b_3)
    layer_activation3 = tf.nn.relu(layer_conv3)
    mean, var = tf.nn.moments(layer_activation3, [0, 1, 2])
    batch_norm3 = tf.nn.batch_normalization(layer_activation3, mean, var, 0, 1, 0)

    # Layer 4
    F_W_4 = tf.get_variable('W4',
                               shape=[3, 3, 384, 384],
                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                          mode='FAN_IN',
                                                                                          uniform=False,
                                                                                          seed=None,
                                                                                          dtype=tf.float32))
    F_b_4 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384]))    
    conv = tf.nn.conv2d(batch_norm3, F_W_4, [1, 1, 1, 1], padding='SAME')
    layer_conv4 = tf.nn.bias_add(conv, F_b_4)
    layer_activation4 = tf.nn.relu(layer_conv4)
    mean, var = tf.nn.moments(layer_activation4, [0, 1, 2])
    batch_norm4 = tf.nn.batch_normalization(layer_activation4, mean, var, 0, 1, 0)

    # Layer 5
    F_W_5 = tf.get_variable('W5',
                               shape=[3, 3, 384, 256],
                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                          mode='FAN_IN',
                                                                                          uniform=False,
                                                                                          seed=None,
                                                                                          dtype=tf.float32))
    F_b_5 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256]))    
    conv = tf.nn.conv2d(batch_norm4, F_W_5, [1, 1, 1, 1], padding='SAME')
    layer_conv5 = tf.nn.bias_add(conv, F_b_5)
    layer_activation5 = tf.nn.relu(layer_conv5)
    mean, var = tf.nn.moments(layer_activation5, [0, 1, 2])
    batch_norm5 = tf.nn.batch_normalization(layer_activation5, mean, var, 0, 1, 0)
    layer_pooling5 = tf.nn.max_pool(batch_norm5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully Connected 6
    layer_flatten = tf.contrib.layers.flatten(layer_pooling5)
    layer_fc1 = tf.layers.dense(layer_flatten, units=4096, activation=tf.nn.relu)

    # Fully Connected 7
    layer_fc2 = tf.layers.dense(layer_fc1, units=4096, activation=tf.nn.relu)

    # prediction
    logits = tf.layers.dense(layer_fc2, units=2, activation=tf.nn.softmax)

    return logits
