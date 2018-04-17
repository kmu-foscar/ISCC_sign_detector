import os
import numpy as np
import cv2
import pickle
from wand.image import Image

sign_image_dir = ['C:/Users/YuJiEun/Desktop/Sign-Classifier-Project/image/sign/0'
    ,'C:/Users/YuJiEun/Desktop/Sign-Classifier-Project/image/sign/1'
    ,'C:/Users/YuJiEun/Desktop/Sign-Classifier-Project/image/sign/2'
    ,'C:/Users/YuJiEun/Desktop/Sign-Classifier-Project/image/sign/3'
    ,'C:/Users/YuJiEun/Desktop/Sign-Classifier-Project/image/sign/4'
    ,'C:/Users/YuJiEun/Desktop/Sign-Classifier-Project/image/sign/5'
    ,'C:/Users/YuJiEun/Desktop/Sign-Classifier-Project/image/sign/6'
    ]
none_image_dir=['C:/Users/YuJiEun/Desktop/Sign-Classifier-Project/image/none']
features=[]
labels=[]
dic = {'features':[],'labels':[]}

for dirname in sign_image_dir:
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            img_color = cv2.imread(dirname+'/'+fname, cv2.IMREAD_COLOR)
            img_shrink = cv2.resize(img_color, (227,227), interpolation=cv2.INTER_AREA)
            #img_trim = img_shrink[5:37,35:35+32]
            features.append(img_shrink)
            labels.append(0)

for dirname in none_image_dir:
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            img_color = cv2.imread(dirname+'/'+fname, cv2.IMREAD_COLOR)
            img_shrink = cv2.resize(img_color, (227,227), interpolation=cv2.INTER_AREA)
            #img_trim = img_shrink[5:37,35:35+32]
            features.append(img_shrink)
            labels.append(1)

dic['features']=features
dic['labels']=labels
f= open('data.p', 'wb')
pickle.dump(dic,f)
f.close

