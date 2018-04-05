import Color_Detector 
import cv2

def main():
    cam = cv2.VideoCapture(1)
    while True:
        ret, img = Color_Detector.operate()
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        
if __name__ == '__main__':
    main()