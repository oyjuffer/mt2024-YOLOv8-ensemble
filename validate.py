from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('YOLOv9c\\1\weights\\best.pt') 
    metrics = model.val(split="test", conf = 0.01, iou = 0.5)
    metrics.box.map
    metrics.box.map50
    metrics.box.map75
    metrics.box.maps



    