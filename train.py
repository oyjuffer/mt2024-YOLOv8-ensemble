from ultralytics import YOLO
import gc
import torch

def train():

    name = "YOLOv9c"
    n = 10

    for i in range(n):

        model = YOLO('yolov9c.yaml').load('yolov9c.pt')
        model.train(data='crystals_2600.yaml', 
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
