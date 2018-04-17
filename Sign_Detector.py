from Alexnet import *
import cv2
import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, (None, 227, 227, 3))
y = tf.placeholder(tf.int64, (None))
logits = Alexnet(x)
inference_operation = tf.argmax(logits, 1)
saver = tf.train.Saver()
sess = tf.Session()


def init() :
    saver.restore(sess, tf.train.latest_checkpoint('./alexnet_checkpoint/'))
def operate(cam):
    ret_val, img = cam.read()
    roi = img[0:226, 414:640]
    images = np.array([img])
    inference_operation = tf.argmax(logits, 1)
    inference_output = sess.run(inference_operation, feed_dict={x: images})
    return inference_output
def shutdown() :
    sess.close()

if __name__ == '__main__' :
    cam = cv2.VideoCapture(1)
    while True :
        operate(cam)    