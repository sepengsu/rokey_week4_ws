from ultralytics import YOLO
import torch


def main():
    model = YOLO(r'C:\Users\na062\Desktop\rokey_week4_ws\project\model\weights\yolo11n.pt')
    # # Train the model
    yaml = r'C:\Users\na062\Desktop\rokey_week4_ws\project\model\data.yaml'
    model.train(data=yaml, epochs=300, batch=16, imgsz=320, device=0, project='training_yolo', name='yolo11n_400img')


if __name__ == '__main__':
    main()
