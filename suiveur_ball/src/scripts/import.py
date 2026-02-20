import ultralytics
from ultralytics import YOLO
model=YOLO("/home/ubuntu/catkin_ws/src/suiveur_ball/suiveur_ball/src/scripts/best_seg.pt", task='segment')
model.export(format='onnx',imgsz=[128,128], task='segment', device='cpu')

