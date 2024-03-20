from ultralytics import YOLO
import torch

images = (["CVAT\images\\02ke_D8_ImagerDefaults_9.jpg", 
           "CVAT\images\\01dd_D12_ImagerDefaults_9.jpg",
           "CVAT\images\\038f_B2_ImagerDefaults_9.jpg"
           ]) 

# 
model_1 = YOLO("models\yolov8n_1\weights\\best.pt")
results_1 = model_1(images, save_txt = True, save_conf = True, project = "ensemble", name = "model_1", exist_ok = True)
del model_1
del results_1
torch.cuda.empty_cache()

# 
model_2 = YOLO("models\yolov8n_2\weights\\best.pt")
results_2 = model_2(images, save_txt = True, save_conf = True, project = "ensemble", name = "model_2", exist_ok = True)
del model_2
del results_2
torch.cuda.empty_cache()

# 
model_3 = YOLO("models\yolov8n_3\weights\\best.pt")
results_3 = model_3(images, save_txt = True, save_conf = True, project = "ensemble", name = "model_3", exist_ok = True)
del model_3
del results_3
torch.cuda.empty_cache()