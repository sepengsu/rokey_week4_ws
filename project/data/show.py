import cv2
import os
import json
file = r"C:\Users\na062\Desktop\week4\project\augmented_data\100.jpg_1.jpg"
json_file = r"C:\Users\na062\Desktop\week4\project\augmented_data\100.jpg_1.jpg.anno.json"
img = cv2.imread(file)
with open(json_file,"r") as f:
    annotations = json.load(f)
for ann in annotations:
    x_min = int(ann["annotation_value"]["x"])
    y_min = int(ann["annotation_value"]["y"])
    x_max = int(x_min + ann["annotation_value"]["width"])
    y_max = int(y_min + ann["annotation_value"]["height"])
    
    print(x_min,y_min,x_max,y_max)
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()