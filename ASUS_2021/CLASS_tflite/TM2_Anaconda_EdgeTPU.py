from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
#import picamera
import cv2

# import tensorflow as tf

from PIL import Image
# from tflite_runtime.interpreter import Interpreter

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


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

  labels = load_labels(args.labels)

  # interpreter = tf.lite.Interpreter(args.model)
  # interpreter = Interpreter(args.model)
  interpreter = make_interpreter(args.model)

  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  #with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
    #camera.start_preview()

  cap = cv2.VideoCapture(0)
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

      # print(labels[label_id],prob)
      cv2.putText(crop_img,labels[label_id] + " " 
        + str(round(prob,3)) + " Inference time=" + str(round(inference_time*1000,2)) + "ms", 
        (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

      show_img=cv2.resize(crop_img,(448,448))
      cv2.imshow('Detecting....',show_img)

    times=times+1
    if (times>1):
      times=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
      key_detect = 1

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
