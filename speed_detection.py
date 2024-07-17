import cv2
import torch
import numpy as np
import os
from collections import deque

print("Script started...")

# Set paths
VIDEO_PATH = r"C:\Users\Asus\Desktop\trafficpt2\output_video.mp4"  # Absolute path to the video file
YOLOV5_FOLDER = r"C:\Users\Asus\Desktop\trafficpt2\yolov5"

# Change working directory to YOLOv5 folder
os.chdir(YOLOV5_FOLDER)
print(f"Current working directory: {os.getcwd()}")

# Print files in the current directory
print("Files in the current directory:")
print(os.listdir('.'))

# Load YOLOv5 model
model = torch.hub.load('.', 'custom', path='yolov5s.pt', source='local')
model = model.cpu()  # Run on CPU if CUDA is causing issues
print("YOLOv5 model loaded successfully.")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}")

# Define the ROI (Region of Interest)
roi_start = height // 2
roi_end = height

# Tracking and speed estimation parameters
track_objects = {}
speed_measurements = {}

# Calibration (adjust based on your video)
pixels_per_meter = 30  # Calibrate this value
distance_meters = 10  # Known distance in the video (in meters)

def calculate_speed(time_sec, distance_meters):
    speed_mps = distance_meters / time_sec
    speed_kmh = speed_mps * 3.6
    return speed_kmh

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_count}")
        break

    frame_count += 1
    if frame_count % 2 != 0:  # Process every second frame
        continue

    # Prepare frame for YOLOv5
    results = model(frame[roi_start:roi_end, :])

    # Extract detections for cars
    car_detections = results.pred[0][results.pred[0][:, 5] == 2]  # Class 2 is 'car'

    if len(car_detections):
        # Apply NMS
        keep = torch.ops.torchvision.nms(car_detections[:, :4], car_detections[:, 4], iou_threshold=0.5)
        car_detections = car_detections[keep]

        for i, det in enumerate(car_detections):
            x1, y1, x2, y2 = det[:4].tolist()
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1) + roi_start, int(y2) + roi_start
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if i not in track_objects:
                track_objects[i] = deque(maxlen=30)  # Store last 30 positions

            track_objects[i].append(center)

            if len(track_objects[i]) > 1:
                # Calculate distance traveled
                pixel_distance = np.sqrt(
                    (track_objects[i][-1][0] - track_objects[i][-2][0]) ** 2 +
                    (track_objects[i][-1][1] - track_objects[i][-2][1]) ** 2
                )
                time_elapsed = 1 / fps

                if pixel_distance > 5:  # Ignore small movements
                    meter_distance = pixel_distance / pixels_per_meter
                    speed = calculate_speed(time_elapsed, meter_distance * (distance_meters / height))

                    if i not in speed_measurements:
                        speed_measurements[i] = []
                    speed_measurements[i].append(speed)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display speed if available
            if i in speed_measurements and len(speed_measurements[i]) > 0:
                avg_speed = np.mean(speed_measurements[i][-10:])  # Average of last 10 measurements
                cv2.putText(frame, f"Speed: {avg_speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Traffic Speed Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
print("Script finished.")
