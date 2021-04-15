#!/bin/bash

cd /home/pi

source ~/.profile
workon cv

cd /home/pi/opencv-face-recognition/
python recognize_rock_btn.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle
