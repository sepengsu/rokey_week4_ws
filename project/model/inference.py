from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from point_rotate import PointSort, PointRotate

IMAGE_FILE_PATH = r"C:\Users\na062\Desktop\week4\project\52.jpg"
ACCESS_KEY = "Vc4OHmAg4o5vg9Mme5nqS6R53Wx4TX1K4xTm61Ti"
CLASS_COUNT = {class_name: 0 for class_name in ["BOOTSEL", "USB", "CHIPSET", "OSCILLATOR", "RASPBERRY PICO", "HOLE"]}
CLASS_COUNT["BOOTSEL"] = 1
CLASS_COUNT["USB"] = 1
CLASS_COUNT["CHIPSET"] = 1
CLASS_COUNT["OSCILLATOR"] = 1
CLASS_COUNT["RASPBERRY PICO"] = 1
CLASS_COUNT["HOLE"] = 4
OBJECT_LIST =["BOOTSEL","USB","CHIPSET","OSCILLATOR","RASPBERRY PICO","HOLE"]
BOX_LIST = ['BOOTSEL','USB','CHIPSET','OSCILLATOR','RASPBERRY PICO','HOLE1','HOLE2','HOLE3','HOLE4']
ABNORMAL=1
NORMAL=0

def dict_to_array(box_dict):
    box_list = []
    for name in BOX_LIST:
        box_list.append(box_dict[name])
    return np.array(box_list)

class Tester:
    def __call__(self,model,img):
        self.model = model
        self.img = img
        return self.test()
    
    def inference(self):
        result = self.model.predict(self.img,device = 'cpu',imgsz=300, conf=0.1,verbose = False)[0]
        result = self.convert(result)
        return result
    
    def test(self):
        results = self.inference()
        class_count = {class_name: 0 for class_name in OBJECT_LIST}
        for obj in results["objects"]:
            class_count[obj["class"]] += 1
        box_dict ={name:None for name in BOX_LIST}
        show_dict = {name:None for name in BOX_LIST}
        hole_num = 0
        for obj in results["objects"]:
            name = obj["class"]
            if name == "HOLE":
                hole_num += 1
                name = f"HOLE{hole_num}"
            x1,y1,x2,y2 = obj['box']
            x,y = int((x1+x2)/2),int((y1+y2)/2)
            box_dict[name] = [x,y]
            show_dict[name] = obj["box"]
        if class_count != CLASS_COUNT:
            return ABNORMAL, show_dict

        ps = PointSort(box_dict['USB'],box_dict['CHIPSET'])
        sortings = ps([box_dict['HOLE1'],box_dict['HOLE2'],box_dict['HOLE3'],box_dict['HOLE4']]) 
        if sortings == -1:
            print("sort error")
            return ABNORMAL, show_dict
        box_dict['HOLE1'],box_dict['HOLE2'],box_dict['HOLE3'],box_dict['HOLE4'] = sortings
        pr = PointRotate(35)
        is_possible = pr(box_dict)
        if is_possible == False:
            print("pos error")
            from detect import show
            show(self.img,results["objects"])
            return ABNORMAL, box_dict
        return NORMAL, show_dict
    
    def convert(self,results):
        c_results = {}
        c_results["objects"] = []
        labels = results.names
        clss = results.boxes.cls.tolist()
        boxes = results.boxes.xyxy.tolist()
        for cls,box in zip(clss,boxes):
            obj = {}
            obj["class"] = labels[int(cls)]
            box = [int(b) for b in box]
            obj["box"] = box
            c_results["objects"].append(obj)
        return c_results

class TestAll(Tester):
    def __init__(self,model,root):
        self.model = model
        self.root = root
        self.img_list = os.listdir(root)
        self.results = []
    
    def __call__(self):
        for img_name in self.img_list:
            self.img = cv2.imread(os.path.join(self.root,img_name))
            result = self.test()
            self.results.append(result)
        return self.results
        
    def __len__(self):
        return len(self.img_list)
    
    def df(self):
        trues = np.zeros(70)
        falses = np.ones(30)
        expected = np.concatenate([trues,falses])
        data = {"labels":expected,"predicts":self.results}
        df = pd.DataFrame(data)
        df.replace({0:"normal",1:"abnormal"},inplace=True)
        return df
    
# def main(image):
#     model = YOLO(r'C:\Users\na062\Desktop\rokey_week4_ws\training_yolo\model2\weights\best.pt')
#     Tester()(model,image)

def main(image):
    model = YOLO(r'C:\Users\na062\Desktop\rokey_week4_ws\training_yolo\model2\weights\best.pt')
    # Tester()(model,image)
    result = TestAll(model,r"C:\Users\na062\Desktop\rokey_week4_ws\test_data")()
    print(result)
    # Tester()(model,image)
if __name__ == '__main__':
    image = cv2.imread(r"C:\Users\na062\Desktop\rokey_week4_ws\test_data\3.jpg")
    main(image)
