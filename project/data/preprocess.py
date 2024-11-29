
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2, os, json
class_mapping = {
    "BOOTSEL": 0,
    "CHIPSET": 1,
    "HOLE": 2,
    "USB": 3,
    "RASPBERRY PICO": 4,
    "OSCILLATOR": 5
}
class_decoder = {v:k for k,v in class_mapping.items()}
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
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    def __call__(self, image, boxes, labels):
        # Convert PIL image to numpy array
        image = np.array(image)

        # Perform augmentation
        augmented = self.augment(image=image, bboxes=boxes, labels=labels)
        image = augmented['image']
        boxes = augmented['bboxes']
        labels = augmented['labels']
        return image, boxes, labels
    
class Saver:
    def __init__(self,path,save_folder_path):
        self.path = path
        self.count = 0
        self.save_folder_path = save_folder_path
    
    def save(self,image,boxes,labels,name):
        self.count += 1
        # PyTorch 텐서를 NumPy 배열로 변환 (채널 순서도 맞춤)

        image = image.cpu().numpy().transpose(1, 2, 0)
        # 이미지 저장
        json_data = self.refine_json(boxes,labels)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{self.save_folder_path}\\{name}_{self.count}.jpg",image)
        with open(f"{self.save_folder_path}\\{name}_{self.count}.jpg.anno.json","w") as f:
            json.dump(json_data,f)
        
    def refine_json(self,boxes,labels):
        refined = []
        for box,label in zip(boxes,labels):
            refined.append({
                "annotation_class": class_decoder[label],
                "annotation_value": {
                    "x": box[0],
                    "y": box[1],
                    "width": box[2],
                    "height": box[3]
                }
            })
        return refined


class Preprocessor:
    def __init__(self,folder_path:str,save_folder_path:str,multiplier:int):
        self.folder_path = folder_path
        self.img_list = os.listdir(folder_path)
        self.augmenter = Augmenter()
        self.saver = Saver(folder_path,save_folder_path)
        self.multiplier = multiplier
    
    def preprocess(self):
        for index,img_name in enumerate(self.img_list):
            img = cv2.imread(os.path.join(self.folder_path,img_name)) # BGR
            boxes, labels = self.get_boxes_labels(img_name)
            data =[]
            for _ in range(self.multiplier):
                augmented_img, augmented_boxes, augmented_labels = self.augmenter(img,boxes,labels)
                data.append((augmented_img,augmented_boxes,augmented_labels))
            for img,boxes,labels in data:
                self.saver.save(img,boxes,labels,img_name)
            if (index+1 % 10) == 0:
                print(f"{index+1} images preprocessed")
            break
        print("Preprocessing done")
    
    def get_boxes_labels(self,img_name):
        with open(os.path.join(self.folder_path,img_name[:-4]+".jpg.anno.json"),"r") as f:
            annotations = json.load(f)
        bboxes = []
        category_id = []
        for ann in annotations:
            x_min = ann["annotation_value"]["x"]
            y_min = ann["annotation_value"]["y"]
            x_max = x_min + ann["annotation_value"]["width"]
            y_max = y_min + ann["annotation_value"]["height"]
            bboxes.append([x_min, y_min, x_max - x_min, y_max - y_min])  # COCO 형식
            category_id.append(class_mapping[ann["annotation_class"]])
        return bboxes, category_id
        
if __name__ == "__main__":
    folder_path = r"C:\Users\na062\Desktop\week4\project\train_data"
    save_folder_path = r"C:\Users\na062\Desktop\week4\project\augmented_data"
    multiplier = 5
    preprocessor = Preprocessor(folder_path,save_folder_path,multiplier)
    preprocessor.preprocess()

    
