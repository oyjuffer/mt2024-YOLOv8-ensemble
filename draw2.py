import cv2

# Read the image
image = cv2.imread('datasets/crystals/images/test/018u_D4_ImagerDefaults_6.png')

# Define the coordinates and dimensions for the primary bounding box (normalized)
x1, y1, w1, h1 = 0.47636916847441696, 0.45628269194375465, 0.6386911972565193, 0.8455452226274673

# Define the coordinates and dimensions for the lightly shaded bounding box (normalized)
x2, y2, w2, h2 = 0.47636916847441696, 0.45628269194375465, 0.692752861040871, 0.8773262192076925

# Get image dimensions
height, width = image.shape[:2]

# Convert normalized coordinates to absolute coordinates
x1_abs = int((x1 - w1 / 2) * width)
y1_abs = int((y1 - h1 / 2) * height)
w1_abs = int(w1 * width)
h1_abs = int(h1 * height)

x2_abs = int((x2 - w2 / 2) * width)
y2_abs = int((y2 - h2 / 2) * height)
w2_abs = int(w2 * width)
h2_abs = int(h2 * height)

# Define the color and thickness for the bounding boxes
colour = (0, 255, 0)  # Green color

# Draw the primary bounding box (transparent)
cv2.rectangle(image, (x1_abs, y1_abs), (x1_abs + w1_abs, y1_abs + h1_abs), colour, 2)

# Draw the lightly shaded bounding box beneath the primary one (transparent)
cv2.rectangle(image, (x2_abs, y2_abs), (x2_abs + w2_abs, y2_abs + h2_abs), colour, 1)  

# Display the image
cv2.imshow('Bounding Box', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
