from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from periphery import GPIO

import argparse
import io
import time
import numpy as np
import cv2

import Adafruit_PCA9685

from PIL import Image
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=2)

servo_min = 150  # Min pulse length out of 4096
servo_max = 600  # Max pulse length out of 4096

# Set Servo 1 to PCA9685 CH0
Servo_1_CH = 0

# Set LED 1 to PCA9685 CH15
LED_1_CH = 15

# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
  pulse_length = 1000000    # 1,000,000 us per second
  pulse_length //= 60       # 60 Hz
  print('{0}us per period'.format(pulse_length))
  pulse_length //= 4096     # 12 bits of resolution
  print('{0}us per bit'.format(pulse_length))
  pulse *= 1000
  pulse //= pulse_length
  pwm.set_pwm(channel, 0, pulse)

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  args = parser.parse_args()

  # Set frequency to 60hz, good for servos.
  pwm.set_pwm_freq(60)

  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model,
    experimental_delegates=[load_delegate('libedgetpu.so.1.0')])

  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  cap = cv2.VideoCapture(1)

  #擷取畫面 寬度 設定為640
  cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
  #擷取畫面 高度 設定為480
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  key_detect = 0
  times=1
  while (key_detect==0):
    ret,image_src =cap.read()

    frame_width=image_src.shape[1]
    frame_height=image_src.shape[0]

    cut_d=int((frame_width-frame_height)/2)
    crop_img=image_src[0:frame_height,cut_d:(cut_d+frame_height)]

    image=cv2.resize(crop_img,(224,224),interpolation=cv2.INTER_AREA)

    start = time.perf_counter()
    if (times==1):
      results = classify_image(interpreter, image)
      label_id, prob = results[0]
      inference_time = time.perf_counter() - start

    cv2.putText(crop_img,
      labels[label_id] + " " + str(round(prob,3)) + 
      " Inference time=" + str(round(inference_time*1000,2)) + "ms", 
      (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
      (0,255,255), 2, cv2.LINE_AA)


    # Servo Control with labels[labels_id]
    if (labels[label_id] == '0 INK'):
      pwm.set_pwm(Servo_1_CH, 0, servo_min)
      pwm.set_pwm(LED_1_CH, 0, 2000)
    elif (labels[label_id] == '1 RPI4_BOX'):
      pwm.set_pwm(Servo_1_CH, 0, servo_max)
      pwm.set_pwm(LED_1_CH, 0, 2000)
    elif (labels[label_id] == '2 OTHER'):
      pwm.set_pwm(Servo_1_CH, 0, int((servo_min + servo_max)/2) )
      pwm.set_pwm(LED_1_CH, 0, 0)

    times=times+1
    if (times>1):
      times=1

    cv2.imshow('Detecting....',crop_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      key_detect = 1

  cap.release()
  cv2.destroyAllWindows()

  # LED 1 OFF
  pwm.set_pwm(LED_1_CH, 0, 0)

if __name__ == '__main__':
  main()
