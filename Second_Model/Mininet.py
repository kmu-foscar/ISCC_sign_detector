### Define your architecture here.

import tensorflow as tf
from tensorflow.contrib.layers import flatten
import time
from sklearn.utils import shuffle

def MiniNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    size = 64
    
    # Convolution and Pooling Layer
    SM_W_1 = tf.Variable(tf.truncated_normal([5, 5, 1, int(size/2)], mean = mu, stddev = sigma), name = 'SM_W_1') # (height, width, input_depth, output_depth)
    SM_b_1 = tf.Variable(tf.zeros(int(size/2)), name = 'SM_b_1') # (output_depth)
    layer_conv1 = tf.nn.bias_add(tf.nn.conv2d(x, SM_W_1, strides=[1, 1, 1, 1], padding='VALID'), SM_b_1)
    layer_activation1 = tf.nn.relu(layer_conv1)
    layer_pooling1 = tf.nn.max_pool(layer_activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolution and Pooling Layer
    SM_W_2 = tf.Variable(tf.truncated_normal([5, 5, int(size/2), size], mean = mu, stddev = sigma), name = 'SM_W_2') # (height, width, input_depth, output_depth)
    SM_b_2 = tf.Variable(tf.zeros(size), name = 'SM_b_2') # (output_depth)
    layer_conv2 = tf.nn.bias_add(tf.nn.conv2d(layer_pooling1, SM_W_2, strides=[1, 1, 1, 1], padding='VALID'), SM_b_2)
    layer_activation2 = tf.nn.relu(layer_conv2)
    layer_pooling2 = tf.nn.max_pool(layer_activation2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    layer_flatten = tf.contrib.layers.flatten(layer_pooling2)
    layer_dropout = tf.nn.dropout(layer_flatten, 0.8)
   
    # Fully Connected Layer
    layer_fc1 = tf.contrib.layers.fully_connected(layer_dropout, int(size*4), tf.nn.relu)

    # Fully Connected Layer
    layer_fc2 = tf.contrib.layers.fully_connected(layer_fc1, int(size*2), tf.nn.relu)

    # Fully Connected Layer
    #layer_fc3 = tf.contrib.layers.fully_connected(layer_fc2, 43, tf.nn.relu)
    layer_fc3 = tf.contrib.layers.fully_connected(layer_fc2, 7, tf.nn.relu)
    logits = layer_fc3

    param_list = [SM_W_1, SM_W_2, SM_b_1, SM_b_2]

    return logits , param_list
