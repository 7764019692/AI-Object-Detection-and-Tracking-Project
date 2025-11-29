# yolo_detector.py

"""
This module wraps the YOLOv8 model using the Ultralytics library.
It provides a simple interface to perform object detection on frames.
"""

from ultralytics import YOLO
import numpy as np
from typing import List, Dict
import config

class YOLODetector:
    def __init__(self, model_path: str = None, conf_threshold: float = None):
        """
        Initialize the YOLO detector with a given model path and confidence threshold.
        """
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.conf_threshold = conf_threshold or config.CONFIDENCE_THRESHOLD
        self.model = YOLO(self.model_path)

    def detect(self, frame) -> List[Dict]:
        """
        Perform object detection on a single frame.

        Args:
            frame (numpy.ndarray): The image frame (BGR) from OpenCV.

        Returns:
            List[Dict]: A list of detections where each detection is a dictionary:
                {
                    "label": str,
                    "confidence": float,
                    "bbox": (x1, y1, x2, y2)
                }
        """
        results = self.model(frame)[0]  # Get first batch result

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            label = self.model.names[cls_id]
            conf = float(box.conf[0].item())

            if conf < self.conf_threshold:
                continue

            # YOLO gives boxes as (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (int(x1), int(y1), int(x2), int(y2))
            })

        return detections
