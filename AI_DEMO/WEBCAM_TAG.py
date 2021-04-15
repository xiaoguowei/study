#!/usr/bin/env python
# coding: utf-8

# # WEBCAM影像加註文字縮放的使用(Python)

import cv2
import time
import argparse
import os

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--tag', help='File path of .tflite file.', required=True)
args = parser.parse_args()

folder = os.path.exists(args.tag)

#判斷結果
if not folder:
    #如果不存在，則建立新目錄
    os.makedirs(args.tag)
    print('資料夾 ' + args.tag + ' 建立成功')

else:
    #如果目錄已存在，則不建立，提示目錄已存在
    print('資料夾 ' + args.tag +' 已存在')

DELAY=float(input('請輸入縮時攝影間隔秒數 (0.1 ~ 3600) ='))

while (DELAY<0.1) or (DELAY>3600) :
    DELAY=float(input('請輸入攝影間隔秒數 (0.1 ~ 3600) ='))

# 選擇攝影機
cap = cv2.VideoCapture(0)

# 設定影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

i=1
PRE_SECOND=time.time()
while(True):
  # 從攝影機擷取一張影像
  ret, frame_src = cap.read()
  
  frame_width=frame_src.shape[1]
  frame_height=frame_src.shape[0]

  cut_d=int((frame_width-frame_height)/2)
  crop_img=frame_src[0:frame_height,cut_d:(cut_d+frame_height)]

  image=cv2.resize(crop_img,(224,224),interpolation=cv2.INTER_AREA)

  # 顯示圖片
  cv2.putText(crop_img,str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' No.'+str(i) , 
    (10,20) , cv2.FONT_HERSHEY_SIMPLEX,  0.6, 
    (0, 255, 255), 1, cv2.LINE_AA)

  show_img=cv2.resize(crop_img,(448,448))

  cv2.imshow('WEBCAM', show_img)
  
  POST_SECOND=time.time()
  if POST_SECOND-PRE_SECOND >= DELAY :
    # cv2.imshow('WEBCAM', show_frame)
    
    cv2.imwrite(args.tag + '/OUT' + str(time.time())+'.jpg',image)
    
    i=i+1
    PRE_SECOND=time.time()
    
  keyValue=cv2.waitKey(1)
  if keyValue & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()
 
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
