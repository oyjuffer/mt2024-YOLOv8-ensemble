import cv2
import os
import json

def draw(predictions_path, image_path, save_path):

    image_names = os.listdir(image_path)

    for image_name in image_names:

        image = cv2.imread(os.path.join(image_path, image_name))
        image_name = os.path.splitext(os.path.basename(image_name))[0]

        with open(os.path.join(predictions_path, f"{image_name}.json"), 'r') as file:
            objects = json.load(file)

        if objects is None:
            continue

        for object in objects:
            label, x, x_std, y, y_std, w, w_std, h, h_std, confidence, confidence_std = object

            height, width = image.shape[:2]
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            # Ensure bounding box stays within image boundaries
            if label == 0:
                colour = (255, 0, 0)  # Red
                object_name = "others"
            elif label == 1:
                colour = (0, 255, )  # Green
                object_name = "clear"
            elif label == 2:
                colour = (0, 0, 255)  # Blue
                object_name = "crystal"
            elif label == 3:
                colour = (255, 255, 0)  # Yellow
                object_name = "precipitate"
            elif label == 4:
                colour = (0, 255, 255)  # Cyan
                object_name = "crystals"
            elif label == 5:
                colour = (255, 0, 255)  # Magenta 
                object_name = "other"
            else:
                colour = (0, 0, 0)  # Default color for unknown label
                object_name = "unknown"

            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

            # Calculate text position
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y2 + 20

            cv2.putText(image, f"{object_name}: {confidence:.2f} ({confidence_std:.2f})", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        # Save the output image

        path = os.path.join(save_path, "images")
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, f"{image_name}.jpg")
        cv2.imwrite(output_path, image)



# DRAW BOUNDING BOXES ON IMAGES
predictions_path = "ensemble_YOLOv9c\output_0.60\\0.25_0.50"
image_path = "datasets\crystals_2600\images\\test"
save_path = "ensemble_YOLOv9c"
draw(predictions_path, image_path, save_path)