import numpy as np


class PathPlanning:

    def __init__(self):
        pass

    def create_centerline(self, longer_boundaries: list, shorter_boundaries: list) -> list:
        max_track_width = 32
        centerline = []

        for i in range(0, len(longer_boundaries) - 1, 5):
            outside_point = np.array(longer_boundaries[i])

            # Find nearest outside lane boundary point to the right lane boundary point
            distances = np.linalg.norm(np.array(shorter_boundaries) - outside_point, axis=1)
            if np.min(distances) > max_track_width:
                continue  # Skip this point if the nearest left lane boundary point is too far away

            nearest_inside_index = np.argmin(distances)
            nearest_inside_point = np.array(shorter_boundaries[nearest_inside_index])

            # Calculate the midpoint between the right and left lane boundary points
            midpoint = (outside_point + nearest_inside_point) / 2
            centerline.append(midpoint.tolist())

        return centerline

    def plan(self, left_lane_boundaries: list, right_lane_boundaries: list) -> tuple[list, list]:
        """
        Plan the trajectory based on the detected lane boundaries.

        args:
            left_lane_boundaries (list): A list of points representing the left lane boundaries.
            right_lane_boundaries (list): A list of points representing the right lane boundaries.

        returns:
            trajectory (list): A list of points representing the planned trajectory.
            curvature (list): The curvature of the planned trajectory.
        """
        if len(left_lane_boundaries) == 0 or len(right_lane_boundaries) == 0:
            return [], []

        # Phase 0: wait until the camera has zoomed in
        distance = np.linalg.norm(np.array(left_lane_boundaries[0]) - np.array(right_lane_boundaries[0]))
        while distance < 15:
            return [], []

        # Phase 1: Generate a centerline trajectory based on the detected lane boundaries
        centerline = self.create_centerline(right_lane_boundaries, left_lane_boundaries)

        return centerline, []