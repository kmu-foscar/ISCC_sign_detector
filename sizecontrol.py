import os
import numpy as np
import cv2
import pickle
from wand.image import Image

dirname = "./own-data"
for root, dirs, files in os.walk(dirname):
    for fname in files:
        print(fname)
        img_color = cv2.imread(dirname+'/'+fname, cv2.IMREAD_COLOR)
        #img_shrink = cv2.resize(img_color, None, fx=0.07, fy=0.07, interpolation=cv2.INTER_AREA)
        img_shrink = cv2.resize(img_color, (32, 32), interpolation=cv2.INTER_AREA)
        #img_trim = img_shrink[5:37,35:35+32]
        cv2.imwrite(dirname+'/'+fname, img_shrink)


