import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class Lane:
    def __init__(self, name, x_range, line_y):
        self.name = name
        self.x_start, self.x_end = x_range
        self.line_y = line_y
        self.count = 0
        self.active_vehicles = set()
        self.counted_vehicles = set()
        self.light_state = "red"

class TrafficSystem:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.lanes = [
            Lane("Left Lane", (0, 213), 300),
            Lane("Middle Lane", (214, 426), 300),
            Lane("Right Lane", (427, 640), 300)
        ]
        self.light_radius = 20
        self.frame_width = 640
        self.frame_height = 480

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            results = self.model.track(
                frame,
                persist=True,
                classes=[2, 3, 5, 7],
                conf=0.5,
                verbose=False
            )

            self.process_detections(frame, results)
            self.update_traffic_lights()  # This now works
            self.draw_interface(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_detections(self, frame, results):
        if results[0].boxes.id is None:
            return

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            for lane in self.lanes:
                if lane.x_start <= cx <= lane.x_end:
                    lane.active_vehicles.add(track_id)
                    
                    if track_id not in lane.counted_vehicles:
                        if abs(cy - lane.line_y) < 10:
                            lane.count += 1
                            lane.counted_vehicles.add(track_id)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break

    # ADD THIS MISSING METHOD
    def update_traffic_lights(self):
        for lane in self.lanes:
            lane.light_state = "green" if len(lane.active_vehicles) > 0 else "red"

    def draw_interface(self, frame):
        # Draw lane boundaries
        cv2.line(frame, (213, 0), (213, self.frame_height), (255, 0, 0), 2)
        cv2.line(frame, (426, 0), (426, self.frame_height), (255, 0, 0), 2)

        info_panel = np.zeros((150, self.frame_width, 3), dtype=np.uint8)
        light_spacing = self.frame_width // len(self.lanes)
        
        for i, lane in enumerate(self.lanes):
            x_pos = (i * light_spacing) + (light_spacing // 2)
            color = (0, 255, 0) if lane.light_state == "green" else (0, 0, 255)
            
            cv2.circle(info_panel, (x_pos, 40), self.light_radius, color, -1)
            cv2.putText(info_panel, lane.name, (x_pos-60, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, f"Count: {lane.count}", (x_pos-60, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        combined = np.vstack([frame, info_panel])
        cv2.imshow('Smart Traffic Control System', combined)

if __name__ == "__main__":
    system = TrafficSystem()
    system.process_video('traffic_video.mp4')