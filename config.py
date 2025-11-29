# config.py

"""
Configuration file for the AI-Driven Real-Time Object Detection and Tracking System.
You can modify paths and parameters here as per your environment.
"""

# Path to YOLOv8 model (can be yolov8n.pt, yolov8s.pt, or a custom model)
YOLO_MODEL_PATH = "yolov8n.pt"  # Make sure this model file is downloaded

# Confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.4

# IOU threshold (can be used later for NMS or tracking tuning)
IOU_THRESHOLD = 0.45

# Path to output CSV log file
DETECTION_LOG_PATH = "detection_logs.csv"

# Video source: 0 = default webcam, or path to video file, or IP camera URL
VIDEO_SOURCE = 0

# Draw options
SHOW_LABELS = True
SHOW_CONFIDENCE = True

# Frame resize (optional for speed). Set to None to use original frame size.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
