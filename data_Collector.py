import Color_Detector
import cv2
import numpy as np
import tensorflow as tf
from numpy import newaxis
from Mininet import *
import csv
import datetime

def main():
    cam = cv2.VideoCapture(1)
    while True:
        ret, img = Color_Detector.operate(cam)
        if ret:
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            dt= datetime.datetime.now()
            imagename = ".\\image10\\7\\" + str(dt.month)+"-"+ str(dt.day)+"-"+ str(dt.hour)+"-"+ str(dt.minute)+"-"+ str(dt.second)+"-"+ str(dt.microsecond)+ ".jpg"
            cv2.imwrite(imagename,img)
            print(imagename)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()