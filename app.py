from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # Using the 'n' (nano) version for fast inference

# Load an image
image_path = "suitcases.jpg"  # Replace with your actual image file
image = cv2.imread(image_path)

# Run YOLO object detection
results = model(image)

# Dictionary to count occurrences of each detected object
object_counts = Counter()

# Define colors for bounding boxes
colors = np.random.randint(0, 255, size=(len(results[0].names), 3), dtype="uint8")

# Draw bounding boxes and count detected objects
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        label = result.names[int(box.cls[0])]  # Object class name
        color = [int(c) for c in colors[int(box.cls[0])]]  # Assign color per class

        # Increase object count
        object_counts[label] += 1

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        # Text background
        text = f"{label} {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)

        # Draw label text
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Display total object count on the image
y_offset = 30
for obj, count in object_counts.items():
    cv2.putText(image, f"{obj}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    y_offset += 30

# Convert image from BGR to RGB for display
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Objects & Quantities")
plt.show()

# Print detected objects & their count
print("Detected Objects:")
for obj, count in object_counts.items():
    print(f"{obj}: {count}")
