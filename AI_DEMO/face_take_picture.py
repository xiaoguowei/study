# USAGE python recognize.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle --image images/adrian.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import RPi.GPIO as GPIO

#======Video Stream======
#import RPi.GPIO as GPIO
from imutils.video import VideoStream
import time

# initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")
#vs = VideoStream(src=0).start()
cap = cv2.VideoCapture(0)
cap.set(3, 360)
cap.set(4, 240)

#======Video Stream======


GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
btn_status = 0

file_path = '/home/pi/logs/'
file_number =0
file_type = '.jpg'

try:
        while True:
                global btn_status,relay_status
                
                ret, frame = cap.read()
                cv2.imshow('frame',frame)
                k=cv2.waitKey(1) & 0xFF
				
                btn_read = GPIO.input(21)
				
                if k == 48:
                        print("break")
                        break
                elif btn_read == 0 & btn_status == 0:
                        file_number = file_number + 1
                        file_name = file_path + str(file_number) + file_type
                        
                        cv2.imwrite(file_name,frame)
                        print("Save Picture")
                        
                        btn_status = 1
                        img = cv2.imread(file_name)
                        cv2.imshow('snapshop',img)

                elif btn_read == 1:
                        btn_status = 0

        cv2.destroyAllWindows()

except KeyboardInterrupt:
        GPIO.cleanup()
        cap.release()
        cv2.destroyAllWindows()
