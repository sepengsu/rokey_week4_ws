import time
import threading
import serial
import requests
import numpy as np
from io import BytesIO
from pprint import pprint
import sys
import cv2
import time
from ultralytics import YOLO
from PIL import Image, ImageTk
from data.edge3 import Cropper

sys.path.append(r"/home/rokey/rokey_week4_ws/project")
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

from tkinter import Tk, Label, Button
import tkinter as tk

class InferenceSystem:
    def __init__(self,ser,model):
        self.ser = ser
        self.model = model
        self.result = None
        self.start_time = None
        self.img = np.zeros((320, 240, 3), dtype=np.uint8)
        self.box_img = np.zeros((320, 240, 3), dtype=np.uint8)
        self.label = None
        self.crooper = Cropper()

    def __call__(self):
        self.main()

    def main(self):
        while 1:
            data = self.ser.read()
            if data ==b"0":
                print("inference start")
                img = get_img()
                start_inference = time.time()
                if img is not None:
                    img = self.crooper(img)
                tester = Tester()
                result,box = tester(self.model, img) # 0: normal, 1: abnormal
                self.result = result
                self.box = box

                self.img = img
                self.box_img = img.copy() # box 구조: [x1,y1,x2,y2]
                for name,box in self.box.items():
                    if box is None:
                        continue
                    x1,y1,x2,y2 = box
                    cv2.rectangle(self.box_img,(x1,y1),(x2,y2),(0,255,0),2)
                print("inference end")
                self.start_time = start_inference
                self.box_img = self.box_img
                self.label = self.labeling()
                self.ser.write(b"1")
    
    def labeling(self):
        box_list = ['BOOTSEL','USB','CHIPSET','OSCILLATOR','RASPBERRY PICO','HOLE1','HOLE2','HOLE3','HOLE4']
        text = ""
        if self.result is None:
            raise Exception("Inference not performed yet")
        if self.result == 0:
            text = "Normal"
            return text
        detected_objects = self.box.keys()
        test = "Anormal: " 
        for box in box_list:
            if self.box[box] is None:
                text +=f"{box} is abnormal\n"
        text = text[:-1]
        return text

            
    def close(self):
        self.ser.close()
        self.result = None
        self.start_time = None
        self.img = None


import tkinter as tk
from tkinter import Button, Label
import cv2
from PIL import Image, ImageTk

class GUI:
    def __init__(self, inference_system):
        self.inference_system = inference_system
        self.inference_system.ser.write(b'0')  # Set initial state
        self.window = tk.Tk()
        self.window.title("GUI")
        self.window.geometry("640x640")
        self.window.resizable(False, False)

        # Schedule the run_inference method to run after the window is initialized
        self.window.after(0, self.run_inference)
        self.main()

    def main(self):
        self.window = tk.Tk()
        self.window.title("GUI")
        self.window.geometry("640x640")
        self.window.resizable(False, False)

        # Create a frame to hold the images
        self.image_frame = tk.Frame(self.window)
        self.image_frame.pack(pady=20)

        # Create labels inside the frame to display images
        self.label1 = Label(self.image_frame)
        self.label1.pack(side="left", padx=10)

        self.label2 = Label(self.image_frame)
        self.label2.pack(side="left", padx=10)

        # Create a frame to hold the defect information
        self.defect_frame = tk.Frame(self.window)
        self.defect_frame.pack(pady=20)

        # Create a label to display defect information
        self.defect_label = Label(self.defect_frame, text="Defect: None", font=("Helvetica", 16))
        self.defect_label.pack()
        # Create a label to display elapsed time
        self.time_label = Label(self.defect_frame, text="Elapsed Time: 0.00 seconds", font=("Helvetica", 16))
        self.time_label.pack()

        # Create buttons for starting and stopping
        self.start_and_stop_button()

        # Schedule image updates every 100 milliseconds (0.1 seconds)
        self.window.after(100, self.update_text)
        self.window.after(100, self.update_images)
        self.window.mainloop()

    def run_inference(self):
        self.inference_system()

    def update_images(self):
        # Update first image
        if self.inference_system.img is not None:
            img = cv2.cvtColor(self.inference_system.img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (320, 240))
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            self.label1.config(image=img_tk)
            self.label1.image = img_tk

        # Update second image
        if self.inference_system.box_img is not None:
            box_img = cv2.cvtColor(self.inference_system.box_img, cv2.COLOR_BGR2RGB)
            box_img = cv2.resize(box_img, (320, 240))
            box_img = Image.fromarray(box_img)
            box_img_tk = ImageTk.PhotoImage(image=box_img)
            self.label2.config(image=box_img_tk)
            self.label2.image = box_img_tk

        # Re-schedule the update_images method after 100 ms
        self.window.after(100, self.update_images)
    
    def update_text(self):
        # Update defect information
        if self.inference_system.label is not None:
            self.defect_label.config(text=self.inference_system.label)
            print(f"Updated defect label: {self.inference_system.label}")

        # Update elapsed time
        if self.inference_system.start_time is not None and self.inference_system.label is not None:
            elapsed_time = time.time() - self.inference_system.start_time
            self.time_label.config(text=f"Elapsed Time: {elapsed_time:.2f} seconds")
            print("end time ")
            self.inference_system.start_time = None
            self.inference_system.label = None

        # Reschedule the update_text method
        self.window.after(100, self.update_text)

    def start_and_stop_button(self):
        # Create a frame to hold the buttons
        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack(pady=20)

        # Create the Start Button inside the frame
        self.start_button = Button(
            self.button_frame, text="Start Belt", width=30, height=2, command=self.start_belt, bg="green", state="normal"
        )
        self.start_button.pack(side="left", padx=10)  # Add horizontal padding between buttons

        # Create the Stop Button inside the frame
        self.stop_button = Button(
            self.button_frame, text="Stop Belt", width=30, height=2, command=self.stop_belt, bg="red", state="disabled"
        )
        self.stop_button.pack(side="left", padx=10)  # Add padding between buttons

    def start_belt(self):
        print("Start Belt")
        self.inference_system.ser.write(b'1')
        # Disable Start button and enable Stop button
        self.stop_button.config(state="normal")

    def stop_belt(self):
        print("Stop Belt")
        self.inference_system.ser.write(b"0")
        # Disable Stop button and enable Start button
        self.start_button.config(state="normal")


if __name__ == "__main__":
    ser = serial.Serial("/dev/ttyACM0", 9600)
    modelpath = '/home/rokey/rokey_week4_ws/training_yolo/yolo11n_400img4/weights/best.pt'
    model = YOLO(modelpath)
    inference_system = InferenceSystem(ser,model)
    gui = GUI(inference_system)
    