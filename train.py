from ultralytics import YOLO
import gc
import torch

def train():

    name = "YOLOv8s-p2"
    n = 10

    for i in range(n):

        model = YOLO('YOLOv8s-p2.yaml').load('YOLOv8s-p2.pt')
        model.train(data='crystals.yaml', 
                            epochs=100,
                            patience=10,
                            imgsz=608,
                            project=f'{name}',
                            name=f'{i+1}',
                            exist_ok = True,
                            seed = i,
                            deterministic = False)
        
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
