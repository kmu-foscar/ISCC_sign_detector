### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from numpy import newaxis
from Mininet import * 
import csv
from wand.image import Image


sign_names = []
with open('signnames.csv') as signname_file:
    signname_reader = csv.DictReader(signname_file)
    sign_names = [row['SignName'] for row in signname_reader]

# test on own images
own_images = np.array([mpimg.imread("./own-data/" + imageName) for imageName in os.listdir("own-data")])

# convert to B/W
own_images_bw = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in own_images])

# use absolute values
own_images_abs = np.array([cv2.convertScaleAbs(image) for image in own_images_bw])

# apply histogram equalization
own_images_hst_eq = np.array([cv2.equalizeHist(image) for image in own_images_abs])

# reshape for conv layer
own_images_reshaped = own_images_hst_eq[..., newaxis]

# normalize range
own_images_normalized = own_images_reshaped - np.mean(own_images_reshaped)

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int64, (None))

logits = MiniNet(x)
inference_operation = tf.argmax(logits, 1)
saver = tf.train.Saver()

with tf.Session() as sess:
    print("Testing {} test images...".format(len(own_images)))
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    inference_output = sess.run(inference_operation, feed_dict={x: own_images_normalized})
    print("Inferred classes:", inference_output)
    l = sess.run(logits, feed_dict={x: own_images_normalized})
    print("logits: ", l)
    #print(logits)#?,43

    count = len(own_images)
    fig, axs = plt.subplots(3, 5, figsize=(15, 10))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()
    for i in range(0, count):
        image = own_images[i]
        evaluated = inference_output[i]

        axs[i].axis('off')
        axs[i].set_title(sign_names[evaluated])
        axs[i].imshow(image)