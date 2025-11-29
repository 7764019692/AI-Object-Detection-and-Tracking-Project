# logger.py

"""
This module handles logging of detection events to a CSV file.
"""

import os
import pandas as pd
from datetime import datetime
import config

class DetectionLogger:
    def __init__(self, csv_path: str = None):
        self.csv_path = csv_path or config.DETECTION_LOG_PATH

        # If file does not exist, create it with header
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                "timestamp",
                "frame_id",
                "object_label",
                "confidence",
                "track_id",
                "x_center",
                "y_center",
                "width",
                "height"
            ])
            df.to_csv(self.csv_path, index=False)

    def log_detection(self, frame_id: int, label: str, confidence: float,
                      track_id: int, bbox):
        """
        Log a single detection event to the CSV file.

        Args:
            frame_id (int): ID of the current frame.
            label (str): Object class label.
            confidence (float): Detection confidence score.
            track_id (int): Object tracking ID.
            bbox (tuple): (x1, y1, x2, y2) bounding box.
        """
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        record = {
            "timestamp": datetime.now().isoformat(),
            "frame_id": frame_id,
            "object_label": label,
            "confidence": confidence,
            "track_id": track_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height
        }

        df = pd.DataFrame([record])
        df.to_csv(self.csv_path, mode="a", header=False, index=False)
