# 🖼️ Real-Time Object Detection using YOLO and OpenCV

This project implements **real-time object detection** using **YOLO (You Only Look Once) and OpenCV** in Python. It detects multiple objects in a webcam stream or video and draws bounding boxes around them.

## 📌 Features
- 🚀 **Real-time object detection** from webcam or video file.
- 🎯 Uses **YOLO (You Only Look Once)** for accurate detection.
- 📦 Draws **bounding boxes** and labels detected objects.
- 🎨 **Random colors** for each detected class.
- 🖥️ Runs on **Windows, macOS, and Linux**.

## 🛠️ Technologies Used
- **Python**
- **OpenCV** (for image processing)
- **YOLOv3** (pre-trained deep learning model)
- **NumPy** (for numerical operations)

## 📂 Project Structure
Object-Detection-YOLO/
│── 1_project code_py.py      # Main Python script for object detection
│── README.md                 # Project documentation
│── yolov3.weights            # YOLOv3 trained weights file (download required)
│── yolov3.cfg                # YOLO model configuration file
│── coco.names                # COCO dataset class labels (80 object classes)


## 🚀 Installation & Setup
### **1️⃣ Install Dependencies**
Ensure you have **Python** installed. Install required libraries using:
```sh
pip install opencv-python numpy

2️⃣ Download YOLO Files
Download the necessary YOLO model files:

yolov3.weights
yolov3.cfg
coco.names
Place them in your project folder.
