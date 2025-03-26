from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load YOLOv8 model globally
model = YOLO("yolov8n.pt")

@app.route('/detect-objects', methods=['POST'])
def detect_objects():
    try:
        # Get image from request
        image_data = request.json.get('image', '')
        
        # Decode base64 image
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run object detection
        results = model(image)
        
        # Dictionary to count occurrences of each detected object
        object_counts = Counter()
        
        # Prepare detection results
        detections = []
        
        # Define colors for bounding boxes
        colors = np.random.randint(0, 255, size=(len(results[0].names), 3), dtype="uint8")
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                label = result.names[int(box.cls[0])]  # Object class name
                color = [int(c) for c in colors[int(box.cls[0])]]  # Assign color per class
                
                # Increase object count
                object_counts[label] += 1
                
                # Store detection details
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'color': color
                })
        
        # Prepare response
        response = {
            'object_counts': dict(object_counts),
            'detections': detections
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
