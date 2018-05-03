import Color_Detector
import cv2
import numpy as np
import tensorflow as tf
from numpy import newaxis
import csv
import datetime

def main():
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if ret_val:
            roi = img[0:240, 400:640]
            img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
            dt= datetime.datetime.now()
            imagename = "../image/none/" + str(dt.month)+"-"+ str(dt.day)+"-"+ str(dt.hour)+"-"+ str(dt.minute)+"-"+ str(dt.second)+"-"+ str(dt.microsecond)+ ".jpg"
            cv2.imwrite(imagename,img)
            print(imagename)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()