import requests
from requests.auth import HTTPBasicAuth
import cv2, os
import numpy as np
import pandas as pd

IMAGE_FILE_PATH = r"C:\Users\na062\Desktop\week4\project\52.jpg"
ACCESS_KEY = "Vc4OHmAg4o5vg9Mme5nqS6R53Wx4TX1K4xTm61Ti"
CLASS_COUNT = {class_name: 0 for class_name in ["BOOTSEL", "USB", "CHIPSET", "OSCILLATOR", "RASPBERRY PICO", "HOLE"]}
CLASS_COUNT["BOOTSEL"] = 1
CLASS_COUNT["USB"] = 1
CLASS_COUNT["CHIPSET"] = 1
CLASS_COUNT["OSCILLATOR"] = 1
CLASS_COUNT["RASPBERRY PICO"] = 1
CLASS_COUNT["HOLE"] = 4


class Tester:
    def __init__(self,folder_path,url):
        self.folder_path = folder_path
        self.img_list = os.listdir(folder_path)
        self.url = url

    def test(self,img):
        _, img_encoded = cv2.imencode(".jpg", img)
        response = requests.post(url=self.url,
            auth=HTTPBasicAuth("kdt2024_1-11", ACCESS_KEY),
            headers={"Content-Type": "image/jpeg"},
            data = img_encoded.tobytes(),
        )

        results = response.json()
        object_list =["BOOTSEL","USB","CHIPSET","OSCILLATOR","RASPBERRY PICO","HOLE"]
        class_count = {class_name: 0 for class_name in object_list}
        # box_list = [{obj["box"]} for obj in results["objects"]]
        for obj in results["objects"]:
            class_count[obj["class"]] += 1

        return 0 if class_count == CLASS_COUNT else 1 # 0: normal, 1: abnormal
    
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
url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/764a674a-99d5-4d4c-bf84-fc0db8b71b5f/inference"
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