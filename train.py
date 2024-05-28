from ultralytics import YOLO

def train():

    name = "YOLOv9c"
    i = 5

    model = YOLO('yolov9c.yaml')
    model.info()
    model.train(data='crystals.yaml', 
                        epochs=100,
                        patience=10,
                        imgsz=608,
                        project=f'{name}',
                        name=f'{i}',
                        exist_ok = True,
                        seed = i,
                        pretrained = False,
                        deterministic = False)
        

if __name__ == '__main__':
    train()
