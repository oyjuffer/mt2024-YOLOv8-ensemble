import torch
import os
import json

import numpy as np
import matplotlib.pyplot as plt
from ultralytics.utils.metrics import DetMetrics, ConfusionMatrix
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.ops import xywh2xyxy

# Define class names
names = {0: 'clustered other', 1: 'clear', 2: 'discrete crystal', 3: 'precipitate', 4: 'clustered crystals', 5: 'discrete other'}

def evaluate_single(predictions_folder, test_folder):

    prediction_files = os.listdir(predictions_folder)
    confusion_matrix = ConfusionMatrix(nc=6, conf=0.01)
    metrics = DetMetrics(plot = True, names=names)
    validator = DetectionValidator()
    stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
    iouv = torch.linspace(0.5, 0.95, 10)

    for prediction_file in prediction_files:

        prediction_file_path = os.path.join(predictions_folder, prediction_file)
        prediction_name = os.path.splitext(prediction_file)[0]

        # Load predictions
        predictions = []
        with open(prediction_file_path, "r") as file:
            # Read each line in the file
            for line in file:
                # Split the line into individual items
                items = line.strip().split()
                
                # Parse each item as needed
                boxes = [float(item) for item in items[1:5]]
                class_label = float(items[0])  # Assuming the first item is the class label
                confidence_class = float(items[5])  # Assuming the last item is the confidence score
                predictions.append(boxes + [confidence_class] + [class_label])
        
        # Load ground truth
        ground_truth = []
        with open(os.path.join(test_folder, prediction_name + ".txt"), "r") as f:
            for line in f:
                line_data = [float(value) for value in line.split()]
                ground_truth.append(line_data)

        predictions_tensor = torch.tensor(predictions)
        ground_truth_tensor = torch.tensor(ground_truth)

        if predictions_tensor.size(0) != 0:
            predicted_boxes = predictions_tensor[:, [0, 1, 2, 3]]
            predicted_boxes = xywh2xyxy(predicted_boxes)
            confidence_class = predictions_tensor[:, [4, 5]]
            detections = torch.cat((predicted_boxes, confidence_class), dim=1)
        else:
            detections = []

        if ground_truth_tensor.size(0) != 0:
            ground_truth_boxes = xywh2xyxy(ground_truth_tensor[:, 1:])
            ground_truth_classes = ground_truth_tensor[:, :1].squeeze(dim=1)
        else:
            ground_truth_boxes = ground_truth_tensor
            ground_truth_classes = ground_truth_tensor


        # PROCESS THE BATCH
        stat = dict(
            conf=torch.zeros(0),
            pred_cls=torch.zeros(0),
            tp=torch.zeros(len(detections), iouv.numel(), dtype=torch.bool),
        )

        stat["target_cls"] = ground_truth_classes

        if len(detections) == 0:
            if len(ground_truth_classes):
                for k in stats.keys():
                    stats[k].append(stat[k])
                    confusion_matrix.process_batch(None, ground_truth_boxes, ground_truth_classes)
            continue

        stat["conf"] = detections[:, 4]
        stat["pred_cls"] = detections[:, 5]

        if len(ground_truth_classes):
            stat["tp"] = validator._process_batch(detections, ground_truth_boxes, ground_truth_classes)
            confusion_matrix.process_batch(detections, ground_truth_boxes, ground_truth_classes)

        for k in stats.keys():
            stats[k].append(stat[k])

    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in stats.items()}
    if len(stats) and stats["tp"].any():
        metrics.process(**stats)
    nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=6)
    
    metrics.confusion_matrix = confusion_matrix

    return metrics

def evaluate_ensemble(predictions_folder, test_folder):

    prediction_files = os.listdir(predictions_folder)
    confusion_matrix = ConfusionMatrix(nc=6, conf=0.01)
    metrics = DetMetrics(plot = True, names=names)
    validator = DetectionValidator()
    stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
    iouv = torch.linspace(0.5, 0.95, 10)

    for prediction_file in prediction_files:

        # LOAD ALL THE PREDICTIONS
        prediction_file_path = os.path.join(predictions_folder, prediction_file)
        prediction_name = os.path.splitext(prediction_file)[0]

        with open(prediction_file_path, "r") as f:
            predictions = json.load(f)
        
        # Load ground truth
        ground_truth = []
        with open(os.path.join(test_folder, prediction_name + ".txt"), "r") as f:
            for line in f:
                line_data = [float(value) for value in line.split()]
                ground_truth.append(line_data)

        predictions_tensor = torch.tensor(predictions)
        ground_truth_tensor = torch.tensor(ground_truth)

        if predictions_tensor.size(0) != 0:
            predicted_boxes = predictions_tensor[:, [1, 3, 5, 7]]
            predicted_boxes = xywh2xyxy(predicted_boxes)
            confidence_class = predictions_tensor[:, [9, 0]]
            detections = torch.cat((predicted_boxes, confidence_class), dim=1)
        else:
            detections = []

        if ground_truth_tensor.size(0) != 0:
            ground_truth_boxes = xywh2xyxy(ground_truth_tensor[:, 1:])
            ground_truth_classes = ground_truth_tensor[:, :1].squeeze(dim=1)
        else:
            ground_truth_boxes = ground_truth_tensor
            ground_truth_classes = ground_truth_tensor


        # PROCESS THE BATCH
        stat = dict(
            conf=torch.zeros(0),
            pred_cls=torch.zeros(0),
            tp=torch.zeros(len(detections), iouv.numel(), dtype=torch.bool),
        )

        stat["target_cls"] = ground_truth_classes

        if len(detections) == 0:
            if len(ground_truth_classes):
                for k in stats.keys():
                    stats[k].append(stat[k])
                    confusion_matrix.process_batch(None, ground_truth_boxes, ground_truth_classes)
            continue

        stat["conf"] = detections[:, 4]
        stat["pred_cls"] = detections[:, 5]

        if len(ground_truth_classes):
            stat["tp"] = validator._process_batch(detections, ground_truth_boxes, ground_truth_classes)
            confusion_matrix.process_batch(detections, ground_truth_boxes, ground_truth_classes)

        for k in stats.keys():
            stats[k].append(stat[k])

    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in stats.items()}
    if len(stats) and stats["tp"].any():
        metrics.process(**stats)
    nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=6)
    
    metrics.confusion_matrix = confusion_matrix

    return metrics

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (tuple): Tuple containing the coordinates of the first bounding box (x, y, width, height).
        box2 (tuple): Tuple containing the coordinates of the second bounding box (x, y, width, height).

    Returns:
        float: The IoU value.
    """
    c1, x1, y1, w1, h1 = box1
    c2, x2, y2, w2, h2 = box2

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
    iou = intersection_area / (union_area + 1e-16)

    return iou

def binning(predictions_folder, test_folder):

    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    bins_conf = [[] for _ in range(len(bins) - 1)]

    samples = []

    for prediction_file in os.listdir(predictions_folder):
        # LOAD ALL THE PREDICTIONS
        prediction_file_path = os.path.join(predictions_folder, prediction_file)
        prediction_name = os.path.splitext(prediction_file)[0]

        with open(prediction_file_path, "r") as f:
            predictions = json.load(f)
        
        # Load ground truth
        ground_truth = []
        with open(os.path.join(test_folder, prediction_name + ".txt"), "r") as f:
            for line in f:
                line_data = [float(value) for value in line.split()]
                ground_truth.append(line_data)

        for p in predictions:

            matched = False
            confidence = p[9]

            for gt in ground_truth:

                if p[0] == gt[0] and iou([p[0], p[1], p[3], p[5], p[7]], gt) > 0.5:
                    samples.append((confidence, p[0], gt[0]))
                    matched = True
                    break

            if not matched:
                samples.append((confidence, p[0], gt[0]))
        
    # bin the samples
    for s in samples:
        for i, bin_start in enumerate(bins):
            bin_end = bins[i + 1]
            if bin_start <= s[0] < bin_end:
                bins_conf[i].append((s[0], s[1], s[2]))
                break
    
    return bins_conf

def reliability(bins):

    ece = 0
    mean_confidence = []
    fraction_positives = []
    total_length = sum(len(bin) for bin in bins)

    for bin in bins:

        if not bin:
            break

        confidence = [conf[0] for conf in bin]
        mean = np.mean(confidence)
        mean_confidence.append(mean)

        positives = 0
        for label in bin:
            if label[1] == label[2]:
                positives += 1

        fraction = positives / (len(bin) + 1e-16)
        fraction_positives.append(fraction)

        ece += len(bin) / total_length * abs(fraction - mean)
            
    return mean_confidence, fraction_positives, ece

# https://docs.ultralytics.com/reference/utils/metrics/

# SINGE METRICS
# results = evaluate_single("single_YOLOv9c_ib\\10\labels", "datasets\icebear\labels\\test")
# print("---RESULTS SINGLE---")
# print("AP@50: ", results.box.ap50)
# print("mAP@50: ", results.box.map50)
# print("AP@50-95: ", results.box.ap)
# print("mAP@50-95: ", results.box.map)
# print("F1: ", results.box.f1)
# print()

# bins = binning("single_YOLOv9c\\1\labels", "datasets\crystals\labels\\test")
# c, f, ece = reliability(bins)

# plt.plot(c, f, linewidth=2, marker='o', markersize=5, markerfacecolor='r')
# plt.plot([0, 1], [0, 1], color='0.7', linestyle='--', label='Perfect Calibration')
# plt.xlabel('Mean Predicted Confidence')
# plt.ylabel('Fraction of Positives')
# plt.title('Reliability Plot')
# plt.text(0.95, 0.05, f'ECE: {ece:.3f}', ha='right', va='bottom', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
# plt.grid(True)
# plt.savefig('reliability_plot_single.png')
# plt.show()


# ENSEMBLE METRICS
# results = evaluate_ensemble("ensemble_YOLOv9c\output\\100.00", "datasets\crystals\labels\\test")
# print("---RESULTS ENSEMBLE---")
# print("AP@50: ", results.box.ap50)
# print("mAP@50: ", results.box.map50)
# print("AP@50-95: ", results.box.ap)
# print("mAP@50-95: ", results.box.map)
# print("F1: ", results.box.f1)


bins = binning("ensemble_YOLOv9c\output\\100.00", "datasets\crystals\labels\\test")
c, f, ece = reliability(bins)

plt.plot(c, f, linewidth=2, marker='o', markersize=5, markerfacecolor='r')
plt.plot([0, 1], [0, 1], color='0.7', linestyle='--', label='Perfect Calibration')
plt.xlabel('Mean Predicted Confidence')
plt.ylabel('Fraction of Positives')
plt.title('Reliability Plot')
plt.text(0.95, 0.05, f'ECE: {ece:.3f}', ha='right', va='bottom', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.grid(True)
plt.savefig('reliability_plot_ensemble.png')
plt.show()