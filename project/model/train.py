from ultralytics import YOLO
import torch

model = YOLO('yolov5n.yaml')

# # Train the model
model.train(data='data.yaml', epochs=3, batch_size=8, imgsz=640,
            device='cpu', project='runs/train', name='exp', exist_ok=True)