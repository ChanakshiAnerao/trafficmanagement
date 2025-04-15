import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Make sure to have the model file

# Video configuration
video_path = 'traffic_video.mp4'  # ← REPLACE WITH YOUR VIDEO PATH
cap = cv2.VideoCapture(video_path)

# Verify video input
if not cap.isOpened():
    print(f"Error: Could not open video source {video_path}")
    exit(1)

# Counting configuration
vehicle_counter = 0
line_y_position = 300
vehicle_classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

# Tracking dictionary
vehicle_data = defaultdict(lambda: {'prev_y': None, 'counted': False})

print("Starting vehicle detection and counting...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]

    # Run YOLOv8 detection
    results = model.track(
        frame,
        persist=True,
        classes=vehicle_classes,
        conf=0.5,
        verbose=False,
        imgsz=320
    )

    # Draw counting line
    cv2.line(frame, (0, line_y_position), (width, line_y_position), (0, 0, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate center point
            cy = (y1 + y2) // 2

            # Update direction tracking
            if vehicle_data[track_id]['prev_y'] is not None:
                prev_y = vehicle_data[track_id]['prev_y']
                direction = "down" if cy > prev_y else "up"
                
                if not vehicle_data[track_id]['counted']:
                    if (direction == "down" and cy >= line_y_position) or \
                       (direction == "up" and cy <= line_y_position):
                        vehicle_counter += 1
                        vehicle_data[track_id]['counted'] = True
                        print(f"Vehicle counted {vehicle_counter}")  # Modified output

            vehicle_data[track_id]['prev_y'] = cy

    # Display simple count on frame
    cv2.putText(frame, f'Count: {vehicle_counter}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Vehicle Counter', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Final count: {vehicle_counter}")