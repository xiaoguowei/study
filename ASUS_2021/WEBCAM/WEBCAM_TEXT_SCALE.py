#!/usr/bin/env python
# coding: utf-8

# # WEBCAM影像加註文字縮放的使用(Python)

import cv2
import time
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--tag', help='File path of .tflite file.', required=True)
args = parser.parse_args()

DELAY=float(input('請輸入縮時攝影間隔秒數 (0.1 ~ 3600) ='))

while (DELAY<0.1) or (DELAY>3600) :
    DELAY=float(input('請輸入攝影間隔秒數 (0.1 ~ 3600) ='))

SCALE=int(input('請輸入畫面大小設定，1:大 2：標準 3:小 ='))

# 選擇第2攝影機
cap = cv2.VideoCapture(1)

# 設定影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

i=1
PRE_SECOND=time.time()
while(True):
  # 從攝影機擷取一張影像
  ret, frame_src = cap.read()
  
  frame=cv2.resize(frame_src,(int(1280/SCALE),int(960/SCALE)))
 
  # 顯示圖片
  cv2.putText(frame,str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' No.'+str(i) , (int(10/SCALE),int(30/SCALE)) , cv2.FONT_HERSHEY_SIMPLEX,  1/SCALE, (0, 255, 255), 1, cv2.LINE_AA)
  cv2.imshow('WEBCAM TEST', frame)
  
  POST_SECOND=time.time()
  if POST_SECOND-PRE_SECOND >= DELAY :
    cv2.imwrite('./Pictures/' + args.tag + '/OUT' + str(time.time())+'.jpg',frame)
    i=i+1
    PRE_SECOND=time.time()
    
  keyValue=cv2.waitKey(1)
  if keyValue & 0xFF == ord('q'):
    break
  elif keyValue & 0xff == ord('1') :
    SCALE=1
  elif keyValue & 0xff == ord('2') :
    SCALE=2
  elif keyValue & 0xff == ord('3') :
    SCALE=3

# 釋放攝影機
cap.release()
 
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
