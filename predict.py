from ultralytics import YOLO

import numpy as np
from torchvision.ops import nms
from ultralytics.utils.ops import xywh2xyxy
import torch
import os
import gc
import json

def predict(model_folder, image_path, confidence_threshold, iou_threshold):
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
        model(image_path, save_txt=True, save_conf=True, project=model_folder + "_predictions", name=model_name, exist_ok = True, conf = confidence_threshold, iou = iou_threshold)

        # release memory
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (tuple): Tuple containing the coordinates of the first bounding box (x, y, width, height).
        box2 (tuple): Tuple containing the coordinates of the second bounding box (x, y, width, height).

    Returns:
        float: The IoU value.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Calculate the area of intersection
    intersection_area = w_intersection * h_intersection

    # Calculate the area of each bounding box
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area + 1e-16

    return iou

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
        for ensemble_name in ensemble_names:
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
                if iou((boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][4]), (boxes[j][1], boxes[j][2], boxes[j][3], boxes[j][4])) > iou_threshold and boxes[i][0] == boxes[j][0] and j not in checked:       
                    checked.append(j)
                    object.append(boxes[j])

            if object:
                objects.append(object)
        
        # ENSEMBLE THE PREVIOUSLY PREDICTED OBJECTS
        output = []
        for obj in objects:
            all_x, all_y, all_w, all_h, all_confidence = [list(t) for t in zip(*obj)][1:]

            avg_x, std_x = np.mean(all_x), np.std(all_x)
            avg_y, std_y = np.mean(all_y), np.std(all_y)
            avg_w, std_w = np.mean(all_w), np.std(all_w)
            avg_h, std_h = np.mean(all_h), np.std(all_h)
            avg_confidence, std_confidence = np.mean(all_confidence), np.std(all_confidence)

            avg_confidence = avg_confidence * min(len(all_confidence), ensemble_count) / ensemble_count
            if avg_confidence > confidence_threshold:
                output.append([obj[0][0], avg_x, std_x, avg_y, std_y, avg_w, std_w, avg_h, std_h, avg_confidence, std_confidence]) 
    
        # NON-MAXIMUM SUPPRESSION (NMS)
        # if output:
        #     output_tensor = torch.tensor([(o[1], o[3], o[5], o[7], o[9]) for o in output])
        #     boxes = xywh2xyxy(output_tensor[:, :4])
        #     scores = output_tensor[:, 4]
        #     selected_indices = nms(boxes, scores, iou_threshold)
        #     output = [output[selected_indices] for selected_indices in selected_indices]

        # DUMP THE ENSEMBLE PREDICTIONS TO A FILE
        path = ensemble_path + "\\" + "ensemble"
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{image_name}.json")
        with open(file_path, "w") as file:
            json.dump(output, file, indent=4)

# GENERATE MODEL AND ENSEMBLE PREDICTIONS
predict("YOLOv9c", "datasets\crystals\images\\test", 0.01, 0.55)
ensemble("YOLOv9c_predictions", "datasets\crystals\images\\test", 10, 0.15, 0.55)