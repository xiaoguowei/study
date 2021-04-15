from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time
import cv2
import json
import random

import pyrealsense2 as rs

import numpy as np

from PIL import Image
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument(
      '--threshold',
      help='Score threshold for detected objects.',
      required=False,
      type=float,
      default=0.3)
  args = parser.parse_args()

  # Configure depth and color streams
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, 30)

  # Start streaming
  pipeline.start(config)

  # Create an align object
  # rs.align allows us to perform alignment of depth frames to others frames
  # The "align_to" is the stream type to which we plan to align depth frames.
  align_to = rs.stream.color
  align = rs.align(align_to)

  labels = load_labels(args.labels)

  # interpreter = tf.lite.Interpreter(args.model)
  # interpreter = Interpreter(args.model)
  interpreter = Interpreter(args.model,
    experimental_delegates=[load_delegate('libedgetpu.so.1')])

  interpreter.allocate_tensors()
  _, model_height, model_width, _ = interpreter.get_input_details()[0]['shape']

  key_detect = 0
  times=1
  while (key_detect==0):

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
      continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    color_width=color_image.shape[1]
    color_height=color_image.shape[0]

    cut_d=int((color_width-color_height)/2)
    crop_color=color_image[0:color_height,cut_d:(cut_d+color_height)]

    crop_width=crop_color.shape[1]
    crop_height=crop_color.shape[0]

    image = cv2.resize(crop_color, (model_width, model_height))

    if (times==1):
      results = detect_objects(interpreter, image, args.threshold)

      # print("Length of results = " ,len(results))

      for num in range(len(results)) :
        label_id=int(results[num]['class_id'])
        box_top=int(results[num]['bounding_box'][0] * crop_height)
        box_left=int(results[num]['bounding_box'][1] * crop_width)
        box_bottom=int(results[num]['bounding_box'][2] * crop_height)
        box_right=int(results[num]['bounding_box'][3] * crop_width)

        point_distance_list=[]
        for point_num in range(100):
          POINT_X = np.random.randint(box_left,box_right)
          POINT_Y = np.random.randint(box_top,box_bottom)
          if (POINT_X > crop_width-1):
            POINT_X = crop_width-1
          if (POINT_X < 0):
            POINT_X = 0
          if (POINT_Y > crop_height-1):
            POINT_Y = crop_height-1
          if (POINT_Y < 0):
            POINT_Y = 0
          # print(POINT_X,',',POINT_Y)
          point_distance_list.append(depth_frame.get_distance(POINT_X, POINT_Y))
        
        point_distance = np.round(np.median(point_distance_list), 3)

        label_text = labels[label_id] +' score=' +str(round(results[num]['score'],3))
        distance_text = str(np.round(point_distance,3)) + 'm'

        cv2.rectangle(crop_color, (box_left,box_top), (box_right,box_bottom), (255,255,0), 3)

        cv2.putText(crop_color,
          label_text, (box_left,box_top+20),
          cv2.FONT_HERSHEY_SIMPLEX,0.6,
          (0,255,255),2,cv2.LINE_AA)

        cv2.putText(crop_color,
          distance_text, (box_left,box_top+40),
          cv2.FONT_HERSHEY_SIMPLEX,0.6, 
          (0,255,255), 2, cv2.LINE_AA)

      show_img = cv2.resize(crop_color,
        (int(crop_width*1),int(crop_height*1)))

      cv2.imshow('Object Detecting....',show_img)

    times=times+1
    if (times>1) :
      times=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
      key_detect = 1

  cv2.destroyAllWindows()
  pipeline.stop()

if __name__ == '__main__':
  main()
