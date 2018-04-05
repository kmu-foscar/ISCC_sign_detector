import os
import numpy as np
import cv2
import pickle
from wand.image import Image
import glob

list_dir = ['./image/0','./image/1','./image/2','./image/3','./image/4','./image/5','./image/6']
features=[]
labels=[]
dic = {'features':[],'labels':[]}

for dirname in list_dir:
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            oldfilename = dirname+'/'+fname
            targetext = "png"
            with Image(filename=oldfilename) as image:
                with image.convert(targetext) as convert:
                    convert.save(filename=oldfilename[0:-4] + "." + targetext)
            os.remove(oldfilename)

