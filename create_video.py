import cv2
import os
import numpy as np

IMAGE_FOLDER = r"C:\Users\Asus\Desktop\trafficpt2\images_video"
VIDEO_NAME = 'output_video.mp4'

images = [img for img in os.listdir(IMAGE_FOLDER) if img.endswith(".png") or img.endswith(".jpg")]
images.sort() 
frame = cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(VIDEO_NAME, fourcc, 30.0, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(IMAGE_FOLDER, image)))
video.release()

print(f"Video created successfully: {VIDEO_NAME}")