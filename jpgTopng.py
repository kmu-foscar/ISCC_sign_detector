import os
import numpy as np
import cv2
import pickle
from wand.image import Image
import glob

list_dir = ['./image20/0','./image20/1','./image20/2','./image20/3','./image20/4','./image20/5','./image20/6','/image20/7']
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

