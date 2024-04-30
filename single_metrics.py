import torch
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.ops import xywh2xyxy

from sklearn.metrics import auc

def evaluate_detection_metrics(predictions_folder, test_folder, model_nr, confidence_threshold, iou_threshold):

    predictions_folder = os.path.join(predictions_folder, str(model_nr), "labels")
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


    precision = tp / (tp + fp)
    precision = np.nan_to_num(precision, nan=1.0)
    recall = tp / (tp + fn)
    recall = np.nan_to_num(recall, nan=1.0)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    # fpr = fp / (fp + tn)
    # specificity = tn / (tn + fp)

    return matrix, tp, fp, fn, precision, recall, f1_score


ensembles_path = "single_YOLOv9c"
test_folder = "datasets\crystals_2600\labels\\test"
n = 9

predictions_folders = os.listdir(ensembles_path)

total_precision = []
total_recall = []

for predictions_folder in predictions_folders:

    predictions_folder_path = os.path.join(ensembles_path, predictions_folder)
    conf_iou = predictions_folder_path.split('\\')[-1].split('_')
    conf = float(conf_iou[0])
    iou = float(conf_iou[1])

    matrix, tp, fp, fn, precision, recall, f1_score = evaluate_detection_metrics(predictions_folder_path, test_folder, n, conf, iou)

    total_precision.append(precision)
    total_recall.append(recall)

    print(predictions_folder_path)
    print(matrix)
    print()
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)

    print()
    print("METRICS")
    print("Precision: \t", precision)
    print("Recall: \t", recall)
    print("F1 Score: \t", f1_score)
    print()

# Precision-Recall Curve
# generate empty lists of the right length
precision_classes = [[] for _ in range(len(total_precision[0]))]
recall_classes = [[] for _ in range(len(total_recall[0]))]

# reoganize such that each element contains all precision and recall per class
for precision, recall in zip(total_precision, total_recall):
    for i, p in enumerate(precision):
        precision_classes[i].append(p)
    for i, r in enumerate(recall):
        recall_classes[i].append(r)

# Class names
class_names = {
    0: 'Clustered Other',
    1: 'Clear',
    2: 'Discrete Crystal',
    3: 'Precipitate',
    4: 'Clustered Crystals',
    5: 'Discrete Other'
}

ap = []
for i in range(6):
    recall_precision_pairs = sorted(zip(recall_classes[i], precision_classes[i]))
    recall, precision = zip(*recall_precision_pairs)

    # Append sentinel values to beginning and end for the graph 
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    area_under_curve = auc(recall, precision)
    ap.append(area_under_curve)

    plt.plot(mrec, mpre, marker='.', label=class_names[i])
    
mAP = np.mean(ap)
print("AP Scores:", ap)
print("mAP Score:", mAP)

# Set labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Show legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()