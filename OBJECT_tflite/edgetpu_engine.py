import argparse
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import cv2
import time, sys

def ReadLabelFile(file_path):
  with open(file_path, 'r', encoding="utf-8") as f:
    lines = f.readlines()

  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()

  return ret

def fps_count(total_frames):
  global last_time, last_frames, fps
  timenow = time.time()

  if(timenow - last_time)>10:
    fps  = (total_frames - last_frames) / (timenow - last_time)
    last_time  = timenow
    last_frames = total_frames

  return fps

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='Path of the detection model.', required=True)
  parser.add_argument(
      '--labels', help='Path of the labels file.')
  args = parser.parse_args()

  engine = DetectionEngine(args.model)
  labels = ReadLabelFile(args.labels) if args.labels else None
  frameid = 0

  while(camera.isOpened()):
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    frame= cv2.resize(frame,(800,480))
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    start = time.perf_counter()

    ans = engine.DetectWithImage(img, threshold=0.4, keep_aspect_ratio=True,
                                 relative_coord=False, top_k=10)
    
    inference_time = time.perf_counter() - start

    if ans:
      for obj in ans:
        box = obj.bounding_box.flatten().tolist()
        x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        lblTXT = "{}({}%)".format(labels[obj.label_id], int(float(obj.score)*100))
        cv2.rectangle( frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText( frame, lblTXT,(x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2)
        fps_rate = fps_count(frameid)

    else:
      print ('No object detected!')

    cv2.putText(frame,"Inference time=" + str(round(inference_time*1000,2)) + "ms",
      (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)

    frameid += 1

    cv2.imshow("FRAME", frame)

    key = cv2.waitKey(1)
    if(key==113):
      sys.exit(0)

if __name__ == '__main__':
  fps = 0
  fps_rate = 0
  start = time.time()
  last_time = time.time()
  last_frames = 0

  camera = cv2.VideoCapture(0)

  main()