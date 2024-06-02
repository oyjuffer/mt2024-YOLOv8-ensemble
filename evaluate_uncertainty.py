import torch
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import ops
from ultralytics.utils.ops import xywh2xyxy

def load(predictions_path, gt_path):

    predictions = []
    gt = []

    for file in os.listdir(predictions_path):
        prediction_file_path = os.path.join(predictions_path, file)
        prediction_name = os.path.splitext(file)[0]
        p = []


        with open(prediction_file_path, "r") as f:
            data = json.load(f)
            predictions.append(data)

        with open(os.path.join(gt_path, prediction_name + ".txt"), "r") as f:
            g = []
            for line in f:
                line_data = [float(value) for value in line.split()]
                g.append(line_data)

        
        gt.append(g)

    return predictions, gt

def fuzzy(pred, std):
    
    output = []

    for i in pred:

        image = []

        for p in i:

            # x, y are coordinates
            # w, h are distance
            label = p[0]
            x, y, w, h = p[1], p[3], p[5], p[7]
            x_std, y_std, w_std, h_std = p[2], p[4], p[6], p[8]
            conf = p[9]

            # add x standard diviations to the original w and h
            w_fuzzy = w + w_std * std
            h_fuzzy = h + h_std * std

            # add the distance betwen points x,y and x,y+std to w and h
            w_fuzzy += abs(x + x_std * std -  x)
            h_fuzzy += abs(y + y_std * std -  y)

            box1 = xywh2xyxy(torch.tensor([[x, y, w, h]], dtype=torch.float))
            box2 = xywh2xyxy(torch.tensor([[x, y, w_fuzzy, h_fuzzy]], dtype=torch.float))
            iou = ops.box_iou(box1, box2).numpy()[0][0]

            p.append(iou)

            image.append(p)
        output.append(image)

    return output


pred, gt = load("YOLOv9c_predictions\ensemble", "datasets\crystals\labels\\test")
pred_ue = fuzzy(pred, 1)

x = [subsub[9] for sublist in pred_ue for subsub in sublist]
y = [subsub[10] for sublist in pred_ue for subsub in sublist]


coefficients = np.polyfit(x, y, 1)
trendline = np.poly1d(coefficients)


plt.scatter(x, y)
plt.plot(x, trendline(x), color='red', linestyle='--', label='Trendline')


plt.xlabel('Confidence')
plt.ylabel('Uncertainty')
plt.title('Scatter Plot Example')

# Display the plot
plt.show()


print()













def binning(predictions_folder, test_folder):

    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    bins_conf = [[] for _ in range(len(bins) - 1)]

    samples = []

    for prediction_file in os.listdir(predictions_folder):
        # LOAD ALL THE PREDICTIONS
        prediction_file_path = os.path.join(predictions_folder, prediction_file)
        prediction_name = os.path.splitext(prediction_file)[0]
        predictions = []

        try:
            with open(prediction_file_path, "r") as f:
                data = json.load(f)
                for element in data:
                    prediction = [element[i] for i in [0, 1, 3, 5, 7, 9]]
                    predictions.append(prediction)
        except:
            with open(prediction_file_path, "r") as f:
                for line in f:
                    line_data = [float(value) for value in line.split()]
                    predictions.append(line_data)

        
        # Load ground truth
        ground_truth = []
        with open(os.path.join(test_folder, prediction_name + ".txt"), "r") as f:
            for line in f:
                line_data = [float(value) for value in line.split()]
                ground_truth.append(line_data)

        for p in predictions:

            matched = False
            confidence = p[5]

            for gt in ground_truth:

                box1 = xywh2xyxy(torch.tensor([p[1:5]], dtype=torch.float))
                box2 = xywh2xyxy(torch.tensor([gt[1:]], dtype=torch.float))
                iou = ops.box_iou(box1, box2).numpy()[0][0]

                if p[0] == gt[0] and iou > 0.55:
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
            continue

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


test_path = "datasets\crystals\labels\\test"
ensemble_path = "YOLOv9c_predictions\ensemble"
model_paths = [
    "YOLOv9c_predictions\\1\labels",
    "YOLOv9c_predictions\\2\labels",
    "YOLOv9c_predictions\\3\labels",
    "YOLOv9c_predictions\\4\labels",
    "YOLOv9c_predictions\\5\labels",
    "YOLOv9c_predictions\\6\labels",
    "YOLOv9c_predictions\\7\labels",
    "YOLOv9c_predictions\\8\labels",
    "YOLOv9c_predictions\\9\labels",
    "YOLOv9c_predictions\\10\labels",
]

# CALIBRATION PLOT
plt.figure(figsize=(10, 8))
for idx, folder in enumerate(model_paths, start=1):
    # Load all the predictions
    bins = binning(folder, test_path)
    c, f, ece = reliability(bins)
    plt.plot(c, f, linewidth=1, marker='o', markersize=3, label=f'Model {idx} (ECE: {ece:.3f})')

bins = binning(ensemble_path, test_path)
c, f, ece = reliability(bins)
plt.plot(c, f, linewidth=2, marker='o', markersize=5, label=f'Ensemble (ECE: {ece:.3f})', color='black')
plt.plot([0, 1], [0, 1], color='0.7', linestyle='--', label='Perfect Calibration')

# Add labels and titles. 
plt.xlabel('Mean Predicted Confidence')
plt.ylabel('Fraction of Positives')
plt.title('Reliability Plot')
plt.grid(True)
plt.legend(loc='best')
plt.savefig('reliability_plot.png')














