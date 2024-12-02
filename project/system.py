import time
import serial
import requests
import numpy
from io import BytesIO
from pprint import pprint
import sys
import cv2
import time
from ultralytics import YOLO
from PIL import Image, ImageTk

sys.path.append("C:\\Users\\na062\\Desktop\\rokey_week4_ws\\project")
from model.inference import Tester

def get_img():
    """Get Image From USB Camera

    Returns:
        numpy.array: Image numpy array
    """

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera Error")
        exit(-1)

    ret, img = cam.read()
    cam.release()

    return img


def crop_img(img, size_dict):
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    img = img[y : y + h, x : x + w]
    return img


from tkinter import Tk, Label, Button

class InferenceSystem:
    def __init__(self,ser,model):
        self.ser = ser
        self.model = model
        self.result = None
        self.start_time = None
        self.img = None

    def __call__(self):
        self.main()

    def main(self):
        while 1:
            data = self.ser.read()
            print(data)
            if data == b"0":
                img = get_img()
                start_inference = time.time()
                crop_info = {"x": 200, "y": 100, "width": 300, "height": 300}

                if crop_info is not None:
                    img = crop_img(img, crop_info)

                tester = Tester()
                result,box = tester(self.model, img) # 0: normal, 1: abnormal
                self.result = result
                self.box = box
                self.start_time = start_inference
                self.img = img
                self.box_img = img.copy() # box 구조: [x1,y1,x2,y2]
                for name,box in self.box.items():
                    x1,y1,x2,y2 = box
                    cv2.rectangle(self.box_img,(x1,y1),(x2,y2),(0,255,0),2)
            else:
                pass
    def close(self):
        self.ser.close()
        self.result = None
        self.start_time = None
        self.img = None


class GUI:
    def __init__(self,inference_system):
        self.inference_system = inference_system
        self.main()

    def main(self):
        self.window = Tk()
        self.window.title("GUI")
        self.window.geometry("640x400")
        self.window.resizable(False, False)

        self.label1 = Label(self.window)
        self.label1.pack(side="left", padx=10, pady=10)

        self.label2 = Label(self.window)
        self.label2.pack(side="right", padx=10, pady=10)

        self.update_images()

    def update_images(self):
        if self.inference_system.img is not None:
            img = cv2.cvtColor(self.inference_system.img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (320, 240))
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            self.label1.config(image=img_tk)
            self.label1.image = img_tk

        if self.inference_system.box_img is not None:
            box_img = cv2.cvtColor(self.inference_system.box_img, cv2.COLOR_BGR2RGB)
            box_img = cv2.resize(box_img, (320, 240))
            box_img = Image.fromarray(box_img)
            box_img_tk = ImageTk.PhotoImage(image=box_img)
            self.label2.config(image=box_img_tk)
            self.label2.image = box_img_tk

        self.window.after(1000, self.update_images)

        self.start_and_stop_buttom()

        self.window.mainloop()

    def start_and_stop_buttom(self):
        self.start_button = Button(self.window, text="Start Belt", width=30, height=5, command=self.start_belt, bg="green")
        self.start_button.pack(side="left", padx=0)

        self.stop_button = Button(self.window, text="Stop Belt", width=30, height=5, command=self.stop_belt, bg="red")
        self.stop_button.pack(side="left", padx=0)

    def start_belt(self):
        print("Start Belt")
        self.ser.write(b"1")
        self.inference_system() # inference system 시행 
    
    def stop_belt(self):
        print("Stop Belt")
        self.ser.write(b"0")

if __name__ == "__main__":
    ser = serial.Serial("/dev/ttyACM0", 9600)
    # modelpath = r"C:\Users\na062\Desktop\rokey_week4_ws\project\model\weights\yolo11n.pt"
    modelpath = 'yolo11n.pt'
    model = YOLO(modelpath)
    inference_system = InferenceSystem(ser,model)
    gui = GUI(inference_system)
    