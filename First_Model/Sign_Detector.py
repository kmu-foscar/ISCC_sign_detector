import Color_Detector
import cv2
import numpy as np
import tensorflow as tf
from numpy import newaxis
from Lenet import *
#import  Sign_Type_Detector
from Sign_Type_Detector import *

import csv

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int64, (None))
param_list=[]
logits, param_list = LeNet(x)
print(param_list)
inference_operation = tf.argmax(logits, 1)
saver = tf.train.Saver(param_list)

def main():
    cam = cv2.VideoCapture(0)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        while True:
            ret_val, img = cam.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if ret_val:
                img = img[0:240, 400:640]
                cv2.imshow('roi', img)
                img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                images = np.array([img])
                images_abs = np.array([cv2.convertScaleAbs(image) for image in images])
                images_hst_eq = np.array([cv2.equalizeHist(image) for image in images_abs])
                images_reshaped = images_hst_eq[..., newaxis]
                images_normalized = images_reshaped - np.mean(images_reshaped)
                inference_output = sess.run(inference_operation, feed_dict={x: images_normalized})

                print("Inferred classes:", inference_output[0])
                print("Inferred classes type:", type(inference_output[0]))
                if(inference_output[0]==1):
                    result = type_Detector(img, sess)
                    print("result :", result)


                #l = sess.run(logits, feed_dict={x: images_normalized})
                #print("logits: ", l)

            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()