import RPi.GPIO as GPIO
import time
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2


# initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")
#vs = VideoStream(src=0).start()
cap = cv2.VideoCapture(0)
cap.set(3, 360)
cap.set(4, 240)

try:
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        #cv2.imwrite('/home/pi/logs/test6.jpg',frame)

        k=cv2.waitKey(1) & 0xFF
        if k == 48:
            print("break")
            break
        elif k== 49:
            cv2.imwrite('/home/pi/logs/test6_1.jpg',frame)
            img = cv2.imread('/home/pi/logs/test6_1.jpg')
            cv2.imshow('snapshop',img)
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
