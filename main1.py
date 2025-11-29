# main.py

"""
Main entry point for the AI-Driven Real-Time Object Detection and Tracking System.
This script:
- Captures frames from webcam or video.
- Runs YOLOv8 detection.
- Tracks objects using a centroid tracker.
- Draws bounding boxes and labels.
- Logs detections to a CSV file.
"""

import cv2
import config
from yolo_detector import YOLODetector
from centroid_tracker import CentroidTracker
from logger import DetectionLogger

def resize_frame(frame):
    """
    Resize frame based on config settings to improve speed.
    """
    if config.FRAME_WIDTH is None or config.FRAME_HEIGHT is None:
        return frame
    return cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

def draw_detections(frame, detections, object_ids_map):
    """
    Draw bounding boxes and labels on the frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        track_id = object_ids_map.get(tuple(det["bbox"]), None)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text_parts = []
        if config.SHOW_LABELS:
            text_parts.append(label)
        if config.SHOW_CONFIDENCE:
            text_parts.append(f"{conf:.2f}")
        if track_id is not None:
            text_parts.append(f"ID:{track_id}")

        text = " ".join(text_parts)
        cv2.putText(frame, text, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    # Initialize components
    detector = YOLODetector()
    tracker = CentroidTracker(max_disappeared=30)
    logger = DetectionLogger()

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_id = 0

    print("Press 'q' to quit the application.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error capturing frame.")
            break

        frame_id += 1
        frame = resize_frame(frame)

        # Detection
        detections = detector.detect(frame)

        # Prepare list of rectangles for tracking
        rects = [det["bbox"] for det in detections]

        # Update tracker
        objects = tracker.update(rects)  # object_id -> centroid

        # Map bounding boxes to track IDs
        # simple nearest centroid mapping
        object_ids_map = {}

        # Build mapping between centroid and bbox for the current frame
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)

            # Find the closest tracked object centroid
            min_dist = None
            assigned_id = None
            for object_id, centroid in objects.items():
                dist = ((centroid[0] - cX) ** 2 + (centroid[1] - cY) ** 2) ** 0.5
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    assigned_id = object_id

            if assigned_id is not None:
                object_ids_map[tuple(det["bbox"])] = assigned_id

                # Log detection
                logger.log_detection(
                    frame_id=frame_id,
                    label=det["label"],
                    confidence=det["confidence"],
                    track_id=assigned_id,
                    bbox=det["bbox"]
                )

        # Draw results
        draw_detections(frame, detections, object_ids_map)

        # Show frame
        cv2.imshow("AI Object Detection & Tracking", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed. Detection logs saved to:", config.DETECTION_LOG_PATH)

if __name__ == "__main__":
    main()
