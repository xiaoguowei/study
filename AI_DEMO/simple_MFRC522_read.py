#!/usr/bin/env python

import RPi.GPIO as GPIO
import time

from mfrc522 import SimpleMFRC522


CD4051_A=33
CD4051_B=35
CD4051_C=37

GPIO.setmode(GPIO.BOARD)
GPIO.setup(CD4051_A, GPIO.OUT)
GPIO.setup(CD4051_B, GPIO.OUT)
GPIO.setup(CD4051_C, GPIO.OUT)

GPIO.output(CD4051_A,GPIO.LOW)
GPIO.output(CD4051_B,GPIO.LOW)
GPIO.output(CD4051_C,GPIO.LOW)


reader = SimpleMFRC522()

try:
  ch=0
  while True:
    id,text = reader.read()
    print(id)
    print(text)
    # time.sleep(1)

except KeyboardInterrupt:
  print('使用者中斷')

GPIO.cleanup()


