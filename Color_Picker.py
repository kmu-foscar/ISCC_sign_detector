import cv2
import numpy as np

def nothing(x) :
    pass

def main():
    cv2.namedWindow('image')
    cv2.createTrackbar('HLo', 'image', 4, 255, nothing)
    cv2.createTrackbar('HUp', 'image', 20, 255, nothing)
    cv2.createTrackbar('SLo', 'image', 144, 255, nothing)
    cv2.createTrackbar('SUp', 'image', 255, 255, nothing)
    cv2.createTrackbar('VLo', 'image', 126, 255, nothing)
    cv2.createTrackbar('VUp', 'image', 255, 255, nothing)
            
    cam = cv2.VideoCapture(1)
    while True:
        ret_val, img = cam.read()
        print(img)
        #cv2.imshow('webcam', img)
        roi = img[0:200, 400:640]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hlo = cv2.getTrackbarPos('HLo', 'image')
        hup = cv2.getTrackbarPos('HUp', 'image')
        slo = cv2.getTrackbarPos('SLo', 'image')
        sup = cv2.getTrackbarPos('SUp', 'image')
        vlo = cv2.getTrackbarPos('VLo', 'image')
        vup = cv2.getTrackbarPos('VUp', 'image')
        
        lower_blue = np.array([hlo, slo, vlo])
        upper_blue = np.array([hup, sup, vup])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(img, img, mask=mask)
        res2 = cv2.medianBlur(res, 15)

        cv2.imshow('img', img)
        #cv2.imshow('mask', mask)
        #cv2.imshow('res', res)
        cv2.imshow('res2', res2)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()