from ultralytics import YOLO
import torch
import torchvision

model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')

model.tune(data='crystals_2600.yaml',
           patience=3,
           epochs=25,
           iterations=100,
           plots=False, 
           save=False, 
           val=False
           )