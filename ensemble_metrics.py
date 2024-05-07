import torch
import os
import json

import numpy as np
from ultralytics.utils.metrics import DetMetrics, ConfusionMatrix
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.ops import xywh2xyxy

# Define class names
names = {0: 'clustered other', 1: 'clear', 2: 'discrete crystal', 3: 'precipitate', 4: 'clustered crystals', 5: 'discrete other'}


def evaluate(predictions_folder, test_folder):

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




# https://docs.ultralytics.com/reference/utils/metrics/

folder = "ensemble_YOLOv8s-p2\output\\1.40"
results = evaluate(folder, "datasets\crystals\labels\\test")

print("---RESULTS---")
print("AP@50: ", results.box.ap50)
print("mAP@50: ", results.box.map50)

print("AP@50-95: ", results.box.ap)
print("mAP@50-95: ", results.box.map)

print("F1: ", results.box.f1)