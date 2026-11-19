import numpy as np


class PathPlanning:
    # Average Track Width measured in pixels
    # Between 20.25 and 24.65 pixels
    MAX_TRACK_WIDTH = 27.5
    MIN_TRACK_WIDTH = 17.5

    def __init__(self):
        pass

    def create_centerline(
        self, 
        longer_boundary: list, 
        shorter_boundary: list
    ) -> list:
        """
        Creates a centerline based on the closest points of each lane boundary.
        The Longer boundary is used as the reference, and the closest point on the shorter boundary is used to create the centerline.
        """
        centerline = []
        widths = []
        shorter_points = np.asarray(shorter_boundary, dtype=float)

        for point in longer_boundary:
            point = np.asarray(point, dtype=float)
            distances = np.linalg.norm(shorter_points - point, axis=1)
            valid_indices = np.flatnonzero(
                (distances <= self.MAX_TRACK_WIDTH)
                & (distances >= self.MIN_TRACK_WIDTH)
            )
            if valid_indices.size == 0:
                continue

            nearest_index = valid_indices[np.argmin(distances[valid_indices])]
            midpoint = (point + shorter_points[nearest_index]) / 2.0
            centerline.append(midpoint.tolist())
            widths.append(distances[nearest_index]) # Store the width for debugging purposes

        # print(f"Average track width: {np.mean(widths):.2f} pixels")
        return centerline

    def plan(
        self, 
        left_lane_boundaries: list, 
        right_lane_boundaries: list
    ) -> tuple[list, list]:
        """
        Plan the trajectory based on the closest midpoint of both lane boundaries.
        """
        if len(left_lane_boundaries) == 0 or len(right_lane_boundaries) == 0:
            return [], []

        centerline = self.create_centerline(
            right_lane_boundaries, left_lane_boundaries
        )

        return centerline, []
