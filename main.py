import Color_Detector 
import cv2
import numpy as np
import tensorflow as tf
from numpy import newaxis
from Mininet import * 
import csv

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int64, (None))
logits = MiniNet(x)
inference_operation = tf.argmax(logits, 1)
saver = tf.train.Saver()

def main():
    cam = cv2.VideoCapture(0)
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        
        while True:
            ret, img = Color_Detector.operate(cam)
            if ret :
                img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                images = np.array([img])
                images_abs = np.array([cv2.convertScaleAbs(image) for image in images])
                images_hst_eq = np.array([cv2.equalizeHist(image) for image in images_abs])
                images_reshaped = images_hst_eq[..., newaxis]
                images_normalized = images_reshaped - np.mean(images_reshaped)
                inference_output = sess.run(inference_operation, feed_dict={x: images_normalized})
                print("Inferred classes:", inference_output)
                l = sess.run(logits, feed_dict={x: images_normalized})
                print("logits: ", l)

            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()