import tensorflow as tf
import sys
import cv2
import time
import win_unicode_console
win_unicode_console.enable()
from PIL import Image
import io
# change this as you see fit
import pickle

#image_path = sys.argv[1]


# Unpersists graph from file
with tf.gfile.FastGFile("work/signs/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

cam = cv2.VideoCapture(0)

with tf.Session() as sess:

    while True:
        ret_val, img = cam.read()

        if ret_val:
            # Loads label file, strips off carriage return
            label_lines = [line.rstrip() for line in tf.gfile.GFile("work/signs/retrained_labels.txt")]

            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            #predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': img})
            predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': img})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()
            time.sleep(.01)
