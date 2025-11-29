🚀 Project Overview

This project implements a real-time object detection and tracking system using deep learning and computer vision techniques.
It combines the power of YOLOv8 for detection and a lightweight Centroid Tracking Algorithm to assign unique IDs to moving objects across frames.

This system can:

Detect multiple objects in real-time

Track each object with a unique ID

Log every detection into a CSV file

Allow further analytics like counting, monitoring, or anomaly detection

It can be deployed on:

CCTV Surveillance Cameras

Autonomous Robots

Intelligent Traffic Systems

Smart Cities and IoT Devices

Industrial Monitoring Systems

✨ Key Features
🔍 Real-Time Object Detection

Uses YOLOv8 model for detecting multiple objects in every video frame.

🏷 Multi-Object Tracking with Unique IDs

Tracks each detected object with a unique ID using centroid tracking.

📄 Automatic Logging System

Saves detection information like:

Timestamp

Frame number

Object label

Confidence score

Tracking ID

Bounding box position

into a CSV file.

🎥 Live Video Feed Processing

Works with:

Laptop webcam

USB camera

CCTV/IP camera

Video files (.mp4, .avi)

⚡ High Processing Speed

Achieves 10–30+ FPS depending on system hardware.

🔧 Modular Code Architecture

Each component is separated into different Python files:

Detection

Tracking

Logging

Configuration

Easy to understand, modify, and expand.

🧠 Technologies Used
Technology	Purpose
Python 3.x	Main programming language
YOLOv8 (Ultralytics)	Deep-learning-based object detection
OpenCV	Video capture, frame processing, drawing
NumPy	Mathematical operations for tracking
Pandas	Storing detection logs
Centroid Tracking Algorithm	Assigning unique IDs to moving objects
🔧 System Architecture
             ┌────────────┐
             │ Video Feed │
             └──────┬─────┘
                    │
               OpenCV Reads Frames
                    │
            ┌───────▼────────┐
            │ YOLOv8 Detector │
            └───────┬────────┘
                    │ Detections
            ┌───────▼─────────┐
            │ Centroid Tracker │
            └───────┬─────────┘
                    │ IDs Assigned
            ┌───────▼──────────┐
            │  Logger (CSV)     │
            └───────┬──────────┘
                    │
            ┌───────▼─────────┐
            │ Display Output   │
            └──────────────────┘

⚙️ Working Principle
1️⃣ Frame Input

OpenCV continuously reads frames from camera/video.

2️⃣ Object Detection

YOLOv8 identifies objects and provides:

bounding boxes

labels

confidence scores

3️⃣ Object Tracking

The CentroidTracker:

computes centroid of each bounding box

matches centroids frame-to-frame

assigns unique IDs

handles disappearing objects

4️⃣ Logging

Every detection is stored in a CSV file.

5️⃣ Output Window

The result is shown in real-time with:

bounding boxes

labels

confidence

tracking ID

📂 Project Structure
AI-Object-Detection-and-Tracking-Project/
│
├── main.py                     # Main pipeline controller
├── yolo_detector.py            # YOLOv8 detection module
├── centroid_tracker.py         # Tracking algorithm
├── logger.py                   # CSV logging system
├── config.py                   # Settings & configurations
├── requirements.txt            # Dependencies
├── detection_logs.csv          # Auto-generated detections
├── README.md                   # Documentation
└── LICENSE                     # MIT License

📥 Installation
1. Clone the repository
git clone https://github.com/7764019692/AI-Object-Detection-and-Tracking-Project.git
cd AI-Object-Detection-and-Tracking-Project

2. Install dependencies
pip install -r requirements.txt

▶️ Running the Project

Simply run:

python main.py


Press Q to quit the window.
