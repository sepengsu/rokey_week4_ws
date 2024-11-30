import requests
from requests.auth import HTTPBasicAuth
import cv2, os
import numpy as np
import pandas as pd
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
    def __init__(self,folder_path,url):
        self.folder_path = folder_path
        self.img_list = os.listdir(folder_path)
        self.url = url

    def inference(self,img):
        _, img_encoded = cv2.imencode(".jpg", img)
        response = requests.post(url=self.url,
            auth=HTTPBasicAuth("kdt2024_1-11", ACCESS_KEY),
            headers={"Content-Type": "image/jpeg"},
            data = img_encoded.tobytes(),
        )
        results = response.json()
        statecode = response.status_code
        if statecode == 200:
            return results
        else:
            raise Exception(f"Error: {statecode}")
        
    def test(self,img):
        results = self.inference(img)
        class_count = {class_name: 0 for class_name in OBJECT_LIST}
        for obj in results["objects"]:
            class_count[obj["class"]] += 1
        if class_count != CLASS_COUNT:
            return ABNORMAL
        box_dict ={name:None for name in BOX_LIST}
        hole_num = 0
        for obj in results["objects"]:
            name = obj["class"]
            if name == "HOLE":
                hole_num += 1
                name = f"HOLE{hole_num}"
            x1,y1,x2,y2 = obj['box']
            x,y = int((x1+x2)/2),int((y1+y2)/2)
            box_dict[name] = [x,y]

        ps = PointSort(box_dict['USB'],box_dict['CHIPSET'])
        sortings = ps([box_dict['HOLE1'],box_dict['HOLE2'],box_dict['HOLE3'],box_dict['HOLE4']]) 
        if sortings == -1:
            print("sort error")
            return ABNORMAL
        box_dict['HOLE1'],box_dict['HOLE2'],box_dict['HOLE3'],box_dict['HOLE4'] = sortings
        pr = PointRotate(35)
        is_possible = pr(box_dict)
        if is_possible == False:
            print("pos error")
            from detect import show
            show(img,results["objects"])
            return ABNORMAL
        return NORMAL

        
    
    def test_all(self):
        result = []
        self.img_list = [f"{i}.jpg" for i in range(1,101)]
        for index,img in enumerate(self.img_list):
            img = cv2.imread(os.path.join(self.folder_path,img))
            result.append(self.test(img))
            if ((index+1) % 10) == 0:
                print(f"{index+1} images tested")
        return result

paths = r"C:\Users\na062\Desktop\week4\project\test_data"
# url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/ed735ff8-c10a-4afb-8a27-aa1c12cc8f73/inference" # 1차
# url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/839bc471-13ab-4df7-aa4d-431eb7ee4bfc/inference"
url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/8d25c903-6bab-46b9-82eb-c4d0e7564f1c/inference"
tester = Tester(paths,url)
trues = np.zeros(70)
falses = np.ones(30)
expected = np.concatenate([trues,falses])
result = tester.test_all()
data = {"labels":expected,"predicts":result}
df = pd.DataFrame(data)
df.replace({0:"normal",1:"abnormal"},inplace=True)
print(df)
df.to_csv("result.csv",index=False)