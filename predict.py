from ultralytics import YOLO

import numpy as np
from torchvision.ops import nms
from ultralytics.utils.ops import xywh2xyxy
import torch
from torchvision import ops
import os
import gc
import json

def predict(project, model_folder, image_path, confidence_threshold, iou_threshold):
    """
    Generate predictions using each ensemble member model.

    Args:
        model_folder (str): Path to the folder containing each ensemble member model.
        image_path (str): Path to the folder containing the images to be predicted.
    """

    # Get the names of the models
    model_names = os.listdir(model_folder)

    # Predict using each model and save to "ensemble_" folder
    for model_name in model_names:
        model = YOLO(os.path.join(model_folder, model_name, "weights", "best.pt"))
        model(image_path, save_txt=True, save_conf=True, project=project, name=model_name, exist_ok = True, conf = confidence_threshold, iou = iou_threshold)

        # release memory
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()

def weighted_mean_and_std(X, C):
    # Convert to numpy arrays for easier mathematical operations
    X = np.array(X)
    C = np.array(C)
    
    # Calculate the weighted mean
    weighted_mean = np.sum(C * X) / np.sum(C)
    
    # Calculate the weighted standard deviation
    variance = np.sum(C * (X - weighted_mean) ** 2) / np.sum(C)
    weighted_std = np.sqrt(variance)
    
    return weighted_mean, weighted_std

def ensemble(ensemble_path, images_path, ensemble_count, confidence_threshold, iou_threshold):
    """
    Combine the predictions of ensemble members for each image.

    Args:
        ensemble_path (str): Path to the folder containing the ensemble members.
        images_path (str): Path to the folder containing the images.
        iou (float): Intersection over Union threshold for combining bounding boxes.
    """

    ensemble_names = os.listdir(ensemble_path)
    image_names = os.listdir(images_path)

    # RUN THE ENSEMBLE FOR EACH IMAGE
    for image_name in image_names:

        image_name = os.path.splitext(os.path.basename(image_name))[0]

        # LOAD THE PREDICTIONS OF EACH ENSEMBLE MEMBER
        boxes = []  # holds the combined predictions of all ensemble members
        for i, ensemble_name in enumerate(ensemble_names):
            if i + 1 > m:
                break

            predictions = os.path.join(ensemble_path, ensemble_name, "labels", image_name + ".txt")

            if os.path.exists(predictions):
                with open(predictions, 'r') as file:
                    for line in file:
                        parts = line.strip().split(' ')
                        label = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        confidence = float(parts[5])
                        boxes.append((label, x, y, width, height, confidence))

        # WEIGHTED BOXES FUSION (WBF)
        objects = []    # holds the combined predictions per each object found in an image 
        checked = []    # holds the indices of the bounding boxes that have been checked
        for i in range(len(boxes)):

            object = [] # holds the combined predictions of the bounding boxes that are part of the same object
            for j in range(i, len(boxes)):

                box1 = xywh2xyxy(torch.tensor([[boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][4]]], dtype=torch.float))
                box2 = xywh2xyxy(torch.tensor([[boxes[j][1], boxes[j][2], boxes[j][3], boxes[j][4]]], dtype=torch.float))
                iou = ops.box_iou(box1, box2).numpy()[0][0]

                if iou > iou_threshold and boxes[i][0] == boxes[j][0] and j not in checked:       
                    checked.append(j)
                    object.append(boxes[j])

            if object:
                objects.append(object)
        
        # ENSEMBLE THE PREVIOUSLY PREDICTED OBJECTS
        output = []
        for obj in objects:
            all_x, all_y, all_w, all_h, all_confidence = [list(t) for t in zip(*obj)][1:]

            avg_x, std_x = weighted_mean_and_std(all_x, all_confidence)
            avg_y, std_y = weighted_mean_and_std(all_y, all_confidence)
            avg_w, std_w = weighted_mean_and_std(all_w, all_confidence)
            avg_h, std_h = weighted_mean_and_std(all_h, all_confidence)
            avg_confidence = np.mean(all_confidence)
            avg_confidence = avg_confidence * min(len(all_confidence), ensemble_count) / ensemble_count

            if avg_confidence > confidence_threshold:
                output.append([obj[0][0], avg_x, std_x, avg_y, std_y, avg_w, std_w, avg_h, std_h, avg_confidence]) 
    
        # NON-MAXIMUM SUPPRESSION (NMS)
        # if output:
        #     output_tensor = torch.tensor([(o[1], o[3], o[5], o[7], o[9]) for o in output])
        #     boxes = xywh2xyxy(output_tensor[:, :4])
        #     scores = output_tensor[:, 4]
        #     selected_indices = nms(boxes, scores, iou_threshold)
        #     output = [output[selected_indices] for selected_indices in selected_indices]

        # DUMP THE ENSEMBLE PREDICTIONS TO A FILE
        path = ensemble_path + "\\" + "ensemble_" + str(m)
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{image_name}.json")
        with open(file_path, "w") as file:
            json.dump(output, file, indent=4)

# GENERATE MODEL AND ENSEMBLE PREDICTIONS
project = "YOLOv9c_predictions_0.01_coco"
conf = 0.01
iou = 0.55

# predict(project, "YOLOv9c", "datasets\coco\images\\test", conf, iou)
for m in range(7, 11):
    ensemble(project, "datasets\coco\images\\test", m, conf, iou)