from ultralytics import YOLO
from PIL import Image, ImageDraw

import numpy as np
import torch
import os
import gc
import json
import cv2


def ensemble():

    model_folders = os.listdir("YOLOv8n")

    for folder in model_folders:
        model = YOLO(os.path.join("YOLOv8n", folder, "weights", "best.pt"))
        results = model("test", save_txt=True, save_conf=True, project="ensemble", name=folder)

        # release memory
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()

def load_objects_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def get_image_file_names(directory):

    files = os.listdir(directory)
    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
    image_names = [os.path.splitext(os.path.basename(file))[0] for file in image_files]

    return image_names

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

    # Handle the case of division by zero
    if union_area == 0:
        return 0

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou

def combine(file_names):

    models = os.listdir("YOLOv8n")

    for name in file_names:

        b = []
        objects = []

        for model in models:
            predictions = os.path.join("ensemble", model, "labels", name + ".txt")

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
                        b.append((label, x, y, width, height, confidence))

        # perform pairwise comparison of bounding boxes
        checked = []
        for i in range(len(b)):

            object = []
            for j in range(i, len(b)):
                if iou((b[i][1], b[i][2], b[i][3], b[i][4]), (b[j][1], b[j][2], b[j][3], b[j][4])) > 0.55 and b[i][0] == b[j][0]:       
                    if j not in checked:
                        checked.append(j)
                        object.append(b[j]) 

            if object:
                objects.append(object)

        # save the output to a file
        path = "ensemble/combined"
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{name}.json")
        with open(file_path, "w") as file:
            json.dump(objects, file, indent=4)

def uncertainty(directory):

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)

                output = []

                for i in range(len(data)):
                    all_x = []
                    all_y = []
                    all_w = []
                    all_h = []
                    all_confidence = []

                    for j in range(len(data[i])):
                        label = data[i][j][0]
                        all_x.append(data[i][j][1])
                        all_y.append(data[i][j][2])
                        all_w.append(data[i][j][3])
                        all_h.append(data[i][j][4])
                        all_confidence.append(data[i][j][5])

                    avg_x = round(np.mean(all_x), 6)
                    std_x = round(np.std(all_x), 6)
                    avg_y = round(np.mean(all_y), 6)
                    std_y = round(np.std(all_y), 6)
                    avg_w = round(np.mean(all_w), 6)
                    std_w = round(np.std(all_w), 6)
                    avg_h = round(np.mean(all_h), 6)
                    std_h = round(np.std(all_h), 6)

                    if len(all_confidence) < 3:
                        all_confidence += [0] * (3 - len(all_confidence))

                    avg_confidence = round(np.mean(all_confidence), 6)
                    std_confidence = round(np.std(all_confidence), 6)

                    output.append([label, avg_x, std_x, avg_y, std_y, avg_w, std_w, avg_h, std_h, avg_confidence, std_confidence])

        # save the output to a file
        path = "ensemble/output"
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{filename}")
        with open(file_path, "w") as file:
            json.dump(output, file, indent=4)

def draw(file_names):

    for name in file_names:

        if os.path.exists(f"test/{name}.jpg"):
            image = cv2.imread(f"test/{name}.jpg")
        elif os.path.exists(f"test/{name}.png"):
            image = cv2.imread(f"test/{name}.png")
        else:
            continue

        json_file = f'ensemble/output/{name}.json'
        objects = load_objects_from_json(json_file)

        for obj in objects:
            label, x, x_std, y, y_std, w, w_std, h, h_std, confidence, confidence_std = obj

            height, width = image.shape[:2]
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            # Ensure bounding box stays within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            if label == 0:
                colour = (255, 0, 0)  # Red
                object_name = "others"
            elif label == 1:
                colour = (0, 255, )  # Green
                object_name = "clear"
            elif label == 2:
                colour = (0, 0, 255)  # Blue
                object_name = "crystal"
            elif label == 3:
                colour = (255, 255, 0)  # Yellow
                object_name = "precipitate"
            elif label == 4:
                colour = (0, 255, 255)  # Cyan
                object_name = "crystals"
            elif label == 5:
                colour = (255, 0, 255)  # Magenta 
                object_name = "other"
            else:
                colour = (0, 0, 0)  # Default color for unknown label
                object_name = "unknown"

            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

            # Calculate text position
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y2 + 20

            cv2.putText(image, f"{object_name}: {confidence:.2f} ({confidence_std:.2f})", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        # Save the output image
        output_directory = "ensemble/images"
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, f"{name}.jpg")
        cv2.imwrite(output_path, image)


ensemble()
file_names = get_image_file_names("test")
combine(file_names)
uncertainty("ensemble/combined")
draw(file_names)
