import os

images = (["CVAT\images\\01dd_D12_ImagerDefaults_9.jpg"]) 

def combine(images):

    models = os.listdir("models")
    ensembles = os.listdir("ensemble")

    for i in images:

        b = []
        l = []
        f = []

        for m in models:
            predictions = os.path.join("ensemble", m, "labels", os.path.basename(i))
            predictions = predictions.replace(".jpg", ".txt")

            with open(predictions, 'r') as file:
                for line in file:
                    parts = line.strip().split(' ')
                    label = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    confidence = float(parts[5])
                    b.append((label, x, y, width, height, confidence))


    print(b)


combine(images)