




from ultralytics import YOLO

# Load a model
model = YOLO("yolov9c.pt")  # load an official model
model = YOLO("YOLOv9c\\1\weights\\best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")