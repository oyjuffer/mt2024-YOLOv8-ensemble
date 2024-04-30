from ultralytics import YOLO

import numpy as np
import torch
import os
import gc

def single_models(model_path, image_path, conf, iou):
    """
    Ensemble function for predicting using multiple models.

    Args:
        model_path (str): Path to the folder containing each ensemble member model.
        image_path (str): Path to the folder containing the images to be predicted.
    """

    # Get the names of the models
    model_names = os.listdir(model_path)

    # Predict using each model and save to "ensemble_" folder
    for model_name in model_names:
        model = YOLO(os.path.join(model_path, model_name, "weights", "best.pt"))
        model(image_path, save_txt=True, save_conf=True, project= "single2_" + model_path + "\\" + "{:.2f}".format(conf) + "_" + "{:.1f}".format(iou), name=model_name, exist_ok = True, conf = conf, iou = iou)

        # release memory
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()

def search(model_path, image_path, conf, iou):

    for c in conf:
        for i in iou:
            single_models(model_path, image_path, c, i)

# Define the model path, image path, confidence, and IoU values
model_path = "YOLOv9c"
image_path = "datasets\crystals_2600\images\\test"
conf = np.arange(0.67, 1.0, 0.01)
iou = [0.5]

search(model_path, image_path, conf, iou)