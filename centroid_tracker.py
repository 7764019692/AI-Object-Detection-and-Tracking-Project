# centroid_tracker.py

"""
A simple centroid-based multi-object tracker.
This tracker assigns IDs to objects based on the
distance between bounding box centroids across frames.
"""

from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared: int = 30):
        """
        Initialize the centroid tracker.

        Args:
            max_disappeared (int): Number of consecutive frames an object
                                   is allowed to be missing before removal.
        """
        # Next unique track ID to assign
        self.next_object_id = 1

        # Dictionary: object_id -> centroid
        self.objects = OrderedDict()

        # How many consecutive frames a given object has disappeared
        self.disappeared = OrderedDict()

        # Max consecutive disappeared frames allowed
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        """
        Register a new object with a new ID.
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        Remove an object ID from our tracking.
        """
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects):
        """
        Update tracked objects based on the new bounding box rectangles.

        Args:
            rects (list of tuples): Each tuple is (x1, y1, x2, y2) for a detection.

        Returns:
            dict: object_id -> centroid
        """
        # If there are no detections, mark existing as disappeared
        if len(rects) == 0:
            # Increase disappeared count for all existing objects
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Deregister if exceeded max_disappeared
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Compute centroids for the new bounding boxes
        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        # If no objects are currently being tracked, register all centroids
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance between each pair of object centroids and input centroids
            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :],
                axis=2
            )

            # Find the smallest value in each row, then sort row indexes
            # based on their minimum values (Hungarian-like greedy approach)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                # Ignore if we have already examined this row or column
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Compute unused rows and columns
            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            # If number of object centroids >= input centroids
            if D.shape[0] >= D.shape[1]:
                # Some objects might have disappeared
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # New objects might have appeared
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects
