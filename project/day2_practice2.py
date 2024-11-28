import requests
from requests.auth import HTTPBasicAuth
import cv2

IMAGE_FILE_PATH = r"C:\Users\na062\Desktop\week4\project\52.jpg"
ACCESS_KEY = "Vc4OHmAg4o5vg9Mme5nqS6R53Wx4TX1K4xTm61Ti"
def test(path):
    image = open(path, "rb").read()
    response = requests.post(
        url="https://suite-endpoint-api-apne2.superb-ai.com/endpoints/cf319288-8d43-4cce-8bb7-690e33a8ff2a/inference",
        auth=HTTPBasicAuth("kdt2024_1-11", ACCESS_KEY),
        headers={"Content-Type": "image/jpeg"},
        data=image,
    )
    image = cv2.imread(IMAGE_FILE_PATH)
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    object_list =["BOOTSEL","USB","CHIPSET","OSCILLATOR","RASPBERRY_PICO","HOLE"]
    color_dict = {obj: color for obj, color in zip(object_list, color_list)}
    class_count = {class_name: 0 for class_name in object_list}
    for obj in object_list:
        color = color_dict[obj]
        class_count[obj] += 1
        x1, y1, x2, y2 = obj["box"]
        start_point = (x1, y1)
        end_point = (x2, y2)
        thickness = 2
        cv2.rectangle(image, start_point, end_point, color, thickness)

        text = obj["class"]
        position = (x1, y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        
    cv2.rectangle(image, (0, 0), (200, 20*(len(class_count)+1)), (255, 255, 255))
    for index, (class_name, count) in enumerate(class_count.items()):
        text = f"{class_name}: {count}"
        position = (20, 20*(index+1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = color_dict[class_name]
        thickness = 2
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    paths = "C://Users//na062//Desktop//week4//project//result_img//" + path.split("//")[-1]
    cv2.imwrite(paths, image)

if __name__ == "__main__":
    import os 
    for path in os.listdir("C://Users//na062//Desktop//week4//project//test_img"):
        test("C://Users//na062//Desktop//week4//project//test_img//" + path)