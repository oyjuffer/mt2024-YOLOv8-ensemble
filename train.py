from ultralytics import YOLO
import gc
import torch

def train(n):

    for i in range(n):

        model = YOLO('yolov8n.yaml').load('yolov8n.pt')
        model.train(data='crystals_2600.yaml', 
                            epochs=100,
                            patience=10,
                            imgsz=608,
                            project='models',
                            name=f'yolov8n_{i+1}',
                            exist_ok = True,
                            seed = i,
                            deterministic = False)
        
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    n = 3
    train(n)

    torch.cuda.empty_cache()
