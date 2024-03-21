from ultralytics import YOLO
import numpy as np
import torch
import os
import gc

images = (["CVAT\images\\02ke_D8_ImagerDefaults_9.jpg", 
           "CVAT\images\\01dd_D12_ImagerDefaults_9.jpg",
           "CVAT\images\\038f_B2_ImagerDefaults_9.jpg"
           ]) 

# load and run all models using images
model_folders = os.listdir("models")
for folder in model_folders:
    model = YOLO(os.path.join("models", folder, "weights", "best.pt"))
    results = model(images, save_txt=True, save_conf=True, project="ensemble", name=folder, exist_ok=True)

    # release memory
    del model
    model = None
    gc.collect()
    torch.cuda.empty_cache()


