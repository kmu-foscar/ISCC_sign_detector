import cv2
import numpy as np

def nothing(x) :
    pass
def operate(cam) :
    ret_val, img = cam.read()
    roi = img[0:240, 400:640]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 50, 0])
    upper_red_1 = np.array([3, 160, 255])
    lower_red_2 = np.array([160, 50, 75])
    upper_red_2 = np.array([180, 160, 255])
    lower_yellow = np.array([10, 100, 0])
    upper_yellow = np.array([25, 180, 255])
    lower_blue = np.array([100, 56, 0])
    upper_blue = np.array([120, 250, 255]) 

    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    res_red = cv2.bitwise_and(roi, roi, mask=mask_red_1)
    res_red_2 = cv2.bitwise_and(roi, roi, mask=mask_red_2)
    cv2.bitwise_or(res_red, res_red_2, res_red)
    res_yellow = cv2.bitwise_and(roi, roi, mask=mask_yellow)
    res_blue = cv2.bitwise_and(roi, roi, mask=mask_blue)
    res = cv2.bitwise_or(res_red, res_yellow)
    cv2.bitwise_or(res, res_blue, res)

    gray = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray ,5,255, cv2.THRESH_BINARY)
    array_sum = thresh.sum()
    cv2.imshow('img', roi)
    if(array_sum > 3000000) :
        print('Sign Detect!')
        return True, gray
    return False, None
    