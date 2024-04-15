from ultralytics import YOLO

model = YOLO('yolov9c.yaml').load('yolov9c.pt')

model.tune(data='crystals_2600.yaml',
           epochs=100,
           patience=10,
           imgsz=608,
           iterations=100,
           project = "YOLOv9c_search",
           name = "search",
           plots=False, 
           save=False, 
           val=False
           )