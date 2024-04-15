from ultralytics import YOLO

model = YOLO("YOLOv9c\\1\weights\\best.pt")
images = (["CVAT\images\\01dd_D12_ImagerDefaults_9.jpg"])

# Run batched inference on a list of images
results = model(images, save_txt = True, save_conf = True, project = "single", exist_ok = True, conf = 0.25)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    # result.show()  # display to screen
    # result.save(filename='result.jpg')  # save to disk