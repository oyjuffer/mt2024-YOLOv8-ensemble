import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define a dictionary of model names and corresponding CSV file paths
model_files = {
    "Model 1": "YOLOv9c/1/results.csv",
    "Model 2": "YOLOv9c/2/results.csv",
    "Model 3": "YOLOv9c/3/results.csv",
    "Model 4": "YOLOv9c/4/results.csv",
    "Model 5": "YOLOv9c/5/results.csv",
    "Model 6": "YOLOv9c/6/results.csv",
    "Model 7": "YOLOv9c/7/results.csv",
    "Model 8": "YOLOv9c/8/results.csv",
    "Model 9": "YOLOv9c/9/results.csv",
    "Model 10": "YOLOv9c/10/results.csv",
    # Add paths for the other models here
}

# Load the metrics/mAP50-95(B) column from each CSV file into a dictionary
box_losses = {}
max_epochs = 0
for model, file_path in model_files.items():
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    loss_values = df["val/cls_loss"].values
    max_epochs = max(max_epochs, len(loss_values))
    box_losses[model] = loss_values

# Pad shorter arrays with NaN values to ensure they all have the same length
for model, loss_values in box_losses.items():
    if len(loss_values) < max_epochs:
        padding = max_epochs - len(loss_values)
        box_losses[model] = np.pad(loss_values, (0, padding), 'constant', constant_values=np.nan)

# Plot the box loss for each model
for i, (model, loss_values) in enumerate(box_losses.items()):
    epochs = range(1, max_epochs + 1)
    plt.plot(epochs, loss_values, label=model, linestyle='--', alpha=0.5, linewidth=1)


# Now, you can calculate the average loss across all models
average_loss = np.nanmean(np.array(list(box_losses.values())), axis=0)


# Plot the average line
plt.plot(epochs, average_loss, label='Average', color='black')

# Add labels and title
plt.xlabel("Epochs")
plt.ylabel("Class Loss")
plt.legend()
plt.grid(True)
plt.ylim(1.2, 3)

# Show the plot
# plt.show()
plt.savefig('val_class_loss.png', dpi = 500)

