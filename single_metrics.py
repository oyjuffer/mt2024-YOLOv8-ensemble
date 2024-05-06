import torch
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.ops import xywh2xyxy

# Class names
class_names = {
    0: 'Clustered Other',
    1: 'Clear',
    2: 'Discrete Crystal',
    3: 'Precipitate',
    4: 'Clustered Crystals',
    5: 'Discrete Other'
}

def evaluate(predictions_folder, test_folder, confidence_threshold, iou_threshold):

    prediction_files = os.listdir(predictions_folder)
    confusion_matrix = ConfusionMatrix(nc=6, conf=confidence_threshold, iou_thres=iou_threshold, task='detect')

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
            detections = None

        if ground_truth_tensor.size(0) != 0:
            ground_truth_boxes = xywh2xyxy(ground_truth_tensor[:, 1:])
            ground_truth_classes = ground_truth_tensor[:, :1]
        else:
            ground_truth_boxes = ground_truth_tensor
            ground_truth_classes = ground_truth_tensor

        confusion_matrix.process_batch(detections, ground_truth_boxes, ground_truth_classes)

    matrix = confusion_matrix.matrix
    tp, fp = confusion_matrix.tp_fp()
    fn = (matrix.sum(0)[:-1] - tp)
    # tn = matrix.sum() - (tp + fp + fn)

    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    # fpr = fp / (fp + tn)
    # specificity = tn / (tn + fp)

    return matrix, tp, fp, fn, precision, recall, f1_score

def mAP50(output, test):
    """	
    Calculate mAP50 for the given predictions and test folder.

    Args:
        output (str): Path to the output folder containing the predictions.
        test (str): Path to the test folder containing the ground truth.
    Returns:

    """
    confidence_threshold = np.arange(0.01, 1.0, 0.01)
    total_precision = []
    total_recall = []
    total_f1_score = []

    for c in confidence_threshold:
            matrix, tp, fp, fn, precision, recall, f1_score = evaluate(output, test, c, 0.5)
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1_score.append(f1_score)

    precision_classes = [[] for _ in range(len(total_precision[0]))]
    recall_classes = [[] for _ in range(len(total_recall[0]))]

    # reoganize such that each element contains all precision and recall per class
    for precision, recall in zip(total_precision, total_recall):
        for i, p in enumerate(precision):
            precision_classes[i].append(p)
        for i, r in enumerate(recall):
            recall_classes[i].append(r)

    ap = []
    plt.figure()
    for i in range(6):
        recall_precision_pairs = sorted(zip(recall_classes[i], precision_classes[i]))
        recall, precision = zip(*recall_precision_pairs)

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        x = np.linspace(0, 1, 101)
        area_under_curve = np.trapz(np.interp(x, mrec, mpre), x)
        ap.append(area_under_curve)

        plt.plot(mrec, mpre, marker='.', label=class_names[i])

    # Set labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    # Show legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.savefig("mAP50.png")
        
    mAP = np.mean(ap)

    return total_precision, total_recall, total_f1_score, ap, mAP

def mAP50_95(output, test):
    """	
    Calculate mAP50 for the given predictions and test folder.

    Args:
        output (str): Path to the output folder containing the predictions.
        test (str): Path to the test folder containing the ground truth.
    Returns:

    """
    iou_threshold = np.arange(0.5, 0.95, 0.05)
    total_precision = []
    total_recall = []
    total_f1_score = []

    for i in iou_threshold:
            matrix, tp, fp, fn, precision, recall, f1_score = evaluate(output, test, 0.01, i)
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1_score.append(f1_score)

    precision_classes = [[] for _ in range(len(total_precision[0]))]
    recall_classes = [[] for _ in range(len(total_recall[0]))]

    # reoganize such that each element contains all precision and recall per class
    for precision, recall in zip(total_precision, total_recall):
        for i, p in enumerate(precision):
            precision_classes[i].append(p)
        for i, r in enumerate(recall):
            recall_classes[i].append(r)

    ap = []
    plt.figure()
    for i in range(6):
        recall_precision_pairs = sorted(zip(recall_classes[i], precision_classes[i]))
        recall, precision = zip(*recall_precision_pairs)

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        x = np.linspace(0, 1, 101)
        area_under_curve = np.trapz(np.interp(x, mrec, mpre), x)
        ap.append(area_under_curve)

        plt.plot(mrec, mpre, marker='.', label=class_names[i])

    # Set labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    # Show legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.savefig("mAP50_95.png")
        
    mAP = np.mean(ap)

    return total_precision, total_recall, total_f1_score, ap, mAP

folder = "ensemble_YOLOv9c\\1\labels"

print("\nmAP50")
total_precision, total_recall, total_f1_score, ap, mAP = mAP50(folder, "datasets\crystals\labels\\test")
print("AP Scores:", ap)
print("mAP Score:", mAP)

print("\nmAP50-95")
total_precision, total_recall, total_f1_score, ap, mAP = mAP50_95(folder, "datasets\crystals\labels\\test")
print("AP Scores:", ap)
print("mAP Score:", mAP)