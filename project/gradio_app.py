import cv2
import gradio as gr
import requests
import numpy as np
from PIL import Image
from requests.auth import HTTPBasicAuth


# 가상의 비전 AI API URL (예: 객체 탐지 API)
TEAM = ""
ACCESS_KEY = ""

IMAGE_FILE_PATH = r"C:\Users\na062\Desktop\week4\project\52.jpg"
ACCESS_KEY = "Vc4OHmAg4o5vg9Mme5nqS6R53Wx4TX1K4xTm61Ti"

def process_image(image):
    # 이미지를 OpenCV 형식으로 변환
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 이미지를 API에 전송할 수 있는 형식으로 변환
    _, img_encoded = cv2.imencode(".jpg", image)

    # API 호출 및 결과 받기 - 실습1
    response = requests.post(
    url="https://suite-endpoint-api-apne2.superb-ai.com/endpoints/cf319288-8d43-4cce-8bb7-690e33a8ff2a/inference",
    auth=HTTPBasicAuth("kdt2024_1-11", ACCESS_KEY),
    headers={"Content-Type": "image/jpeg"},
    data=img_encoded.tobytes(),
    )
    # API 결과를 바탕으로 박스 그리기 - 실습2
    results = response.json()
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    object_list =["BOOTSEL","USB","CHIPSET","OSCILLATOR","RASPBERRY PICO","HOLE"]
    color_dict = {obj: color for obj, color in zip(object_list, color_list)}
    class_count = {class_name: 0 for class_name in object_list}
    print(results)
    for obj in results["objects"]:
        color = color_dict[obj["class"]]
        class_count[obj["class"]] += 1
        x1, y1, x2, y2 = obj["box"]
        start_point = (x1, y1)
        end_point = (x2, y2)
        thickness = 2
        cv2.rectangle(image, start_point, end_point, color, thickness)

        text = obj["class"] + f"({obj['score']:.2f})"
        position = (x1, y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    cv2.rectangle(image, (0, 0), (200, 20*(len(class_count)+1)), (255, 255, 255), -1)
    for index, (class_name, count) in enumerate(class_count.items()):
        text = f"{class_name}: {count}"
        position = (20, 20*(index+1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = color_dict[class_name]
        thickness = 2
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    # BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


# Gradio 인터페이스 설정

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs="image",
    title="Vision AI Object Detection",
    description="Upload an image to detect objects using Vision AI.",
)

# 인터페이스 실행
iface.launch(share=True)
