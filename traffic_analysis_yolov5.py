import torch
import os
from PIL import Image

IMAGE_FOLDER = r"C:\Users\Asus\Desktop\trafficpt2\images"
YOLOV5_FOLDER = r"C:\Users\Asus\Desktop\trafficpt2\yolov5"
os.chdir(YOLOV5_FOLDER)

# Load the YOLOv5 model
model = torch.hub.load('.', 'custom', path='yolov5s.pt', source='local')


vehicle_counts = {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0,
    "bicycle": 0
}

def detect_vehicles(image_path):

    img = Image.open(image_path)

    results = model(img)

    predictions = results.pandas().xyxy[0]  
    
    for i in range(len(predictions)):
        label = predictions.iloc[i]['name']
        if label in vehicle_counts:
            vehicle_counts[label] += 1

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        detect_vehicles(image_path)

print("Vehicle Counts:")
for vehicle, count in vehicle_counts.items():
    print(f"{vehicle.capitalize()}: {count}")