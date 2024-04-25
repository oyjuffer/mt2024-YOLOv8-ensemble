from ultralytics import YOLO

model = YOLO("YOLOv9c\\1\weights\\best.pt")
images = (["CVAT\images\\03cl_H2_ImagerDefaults_9.jpg"])


# Run batched inference on a list of images

# https://docs.ultralytics.com/usage/cfg/#predict-settings

results = model(images, save_txt = True, save_conf = True, project = "single", exist_ok = True, conf = 0.01, iou = 1, agnostic_nms = False)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    # result.show()
    # result.save(filename='result.jpg')