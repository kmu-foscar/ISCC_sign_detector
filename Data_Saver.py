import os
import numpy as np
import cv2
import pickle
from wand.image import Image

list_dir = ['./image10/0','./image10/1','./image10/2','./image10/3','./image10/4','./image10/5','./image10/6']

features=[]
labels=[]
dic = {'features':[],'labels':[]}

for dirname in list_dir:
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            img_color = cv2.imread(dirname+'/'+fname, cv2.IMREAD_COLOR)
            img_shrink = cv2.resize(img_color, (32,32), interpolation=cv2.INTER_AREA)
            #img_trim = img_shrink[5:37,35:35+32]
            features.append(img_shrink)
            labels.append(int(dirname[-1]))
dic['features']=features
dic['labels']=labels
f= open('data.p', 'wb')
pickle.dump(dic,f)
f.close

