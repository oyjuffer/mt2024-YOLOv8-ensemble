from ultralytics.utils.benchmarks import benchmark

if __name__ == '__main__':
    benchmark(model='YOLOv9c\\7\weights\\best.pt', data='crystals_2600.yaml', imgsz=608, half=False, device=0)