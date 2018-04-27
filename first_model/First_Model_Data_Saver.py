import os
import numpy as np
import cv2
import pickle

sign_image_dir = ['../image/sign/0'
    ,'../image/sign/1'
    ,'../image/sign/2'
    ,'../image/sign/3'
    ,'../image/sign/4'
    ,'../image/sign/5'
    ,'../image/sign/6'
    ]
none_image_dir=['../image/none']
features=[]
labels=[]
dic = {'features':[],'labels':[]}

for dirname in sign_image_dir:
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            img_color = cv2.imread(dirname+'/'+fname, cv2.IMREAD_COLOR)
            img_shrink = cv2.resize(img_color, (32 ,32), interpolation=cv2.INTER_AREA)
            #gray = cv2.cvtColor(img_shrink, cv2.COLOR_BGR2GRAY)

            #img_trim = img_shrink[5:37,35:35+32]
            features.append(img_shrink)
            labels.append(1)

for dirname in none_image_dir:
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            img_color = cv2.imread(dirname+'/'+fname, cv2.IMREAD_COLOR)
            img_shrink = cv2.resize(img_color, (32,32), interpolation=cv2.INTER_AREA)
            #gray = cv2.cvtColor(img_shrink, cv2.COLOR_BGR2GRAY)

            #img_trim = img_shrink[5:37,35:35+32]
            features.append(img_shrink)
            labels.append(0)

dic['features']=features
dic['labels']=labels
f= open('data.p', 'wb')
pickle.dump(dic,f)
f.close

