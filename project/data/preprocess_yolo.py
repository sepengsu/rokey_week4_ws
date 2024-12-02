import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os
import json
# 데이터 증강 클래스
class Augmenter:
    def __init__(self):
        self.augment = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.RandomSizedBBoxSafeCrop(224, 224, p=0.5),
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    def __call__(self, image, boxes, labels):
        # 이미지 증강 수행
        augmented = self.augment(image=image, bboxes=boxes, labels=labels)
        return augmented['image'], augmented['bboxes'], augmented['labels']

# 데이터 저장 클래스
class Saver:
    def __init__(self, save_image_folder_path: str, save_txt_folder_path: str):
        self.count = 0
        self.save_image_folder_path = save_image_folder_path
        self.save_txt_folder_path = save_txt_folder_path
    
    def save(self, image, boxes, labels, name):
        self.count += 1

        # PyTorch 텐서를 NumPy 배열로 변환
        image = image.cpu().numpy().transpose(1, 2, 0)
        # 이미지 저장
        cv2.imwrite(f"{self.save_image_folder_path}\\{name[:-4]}_{self.count}.jpg", image)

        # YOLO 형식으로 텍스트 저장
        with open(f"{self.save_txt_folder_path}\\{name[:-4]}_{self.count}.txt", "w") as f:
            for box, label in zip(boxes, labels):
                f.write(f"{label} {box[0]} {box[1]} {box[2]} {box[3]}\n")

# 데이터 전처리 클래스
class Preprocessor:
    def __init__(self, folder_image_path: str, folder_txt_path:str,save_image_folder_path: str, save_txt_folder_path: str, multiplier: int):
        self.folder_image_path = folder_image_path
        self.folder_txt_path = folder_txt_path
        self.img_list = os.listdir(folder_image_path)
        self.augmenter = Augmenter()
        self.saver = Saver(save_image_folder_path, save_txt_folder_path)
        self.multiplier = multiplier
    
    def preprocess(self):
        for index, img_name in enumerate(self.img_list):
            if not img_name.endswith(".jpg"):
                continue  # JPEG 파일만 처리
            img_path = os.path.join(self.folder_image_path, img_name)
            img = cv2.imread(img_path)  # BGR
            boxes, labels = self.get_boxes_labels(img_name)
            
            data = []
            for _ in range(self.multiplier):
                augmented_img, augmented_boxes, augmented_labels = self.augmenter(img, boxes, labels)
                data.append((augmented_img, augmented_boxes, augmented_labels))
            
            for img, boxes, labels in data:
                self.saver.save(img, boxes, labels, img_name)
            if (index + 1) % 10 == 0:
                print(f"{index + 1} images preprocessed")
        print("Preprocessing done")
    
    def get_boxes_labels(self, img_name):
        # YOLO 형식의 텍스트 파일 읽기
        txt_path = os.path.join(self.folder_txt_path, img_name.replace(".jpg", ".txt"))
        bboxes = []
        category_id = []
        
        with open(txt_path, "r") as f:
            annotations = f.readlines()
        
        for ann in annotations:
            data = ann.strip().split()
            class_id = int(data[0])
            x_center = float(data[1])
            y_center = float(data[2])
            width = float(data[3])
            height = float(data[4])
            bboxes.append([x_center, y_center, width, height])
            category_id.append(class_id)
        
        return bboxes, category_id

# 메인 함수
if __name__ == "__main__":
    folder_image_path = r"C:\Users\na062\Desktop\rokey_week4_ws\project\data\images\train"
    folder_txt_path = r"C:\Users\na062\Desktop\rokey_week4_ws\project\data\labels\train"
    save_image_folder_path = r"C:\Users\na062\Desktop\rokey_week4_ws\project\augmented_data\images"
    save_txt_folder_path = r"C:\Users\na062\Desktop\rokey_week4_ws\project\augmented_data\labels"
    multiplier = 5
    preprocessor = Preprocessor(folder_image_path, folder_txt_path, save_image_folder_path,\
                                 save_txt_folder_path, multiplier)
    preprocessor.preprocess()
