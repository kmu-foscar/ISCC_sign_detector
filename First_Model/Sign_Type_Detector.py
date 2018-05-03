import cv2
import numpy as np
import tensorflow as tf
from Second_Model import Mininet
from numpy import newaxis

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int64, (None))
param_list=[]
logits, param_list = Mininet.MiniNet(x)
#print(param_list)
inference_operation = tf.argmax(logits, 1)
saver1 = tf.train.Saver(param_list)
init = tf.global_variables_initializer()

def type_Detector(img, session):


    with session.as_default() as sess:
    #with tf.Session() as sess:

        sess.run(init)
        saver1.restore(sess, tf.train.latest_checkpoint('..\Second_Model\.'))
        images = np.array([img])
        images_abs = np.array([cv2.convertScaleAbs(image) for image in images])
        images_hst_eq = np.array([cv2.equalizeHist(image) for image in images_abs])
        images_reshaped = images_hst_eq[..., newaxis]
        images_normalized = images_reshaped - np.mean(images_reshaped)
        inference_output = sess.run(inference_operation, feed_dict={x: images_normalized})
        result = inference_output[0]
    return result