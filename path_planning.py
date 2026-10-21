import numpy as np


class PathPlanning:

    def __init__(self):
        pass

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
        max_distance = 35

        index_left = 0
        index_right = 0
        centerline = []
        while index_left < len(left_lane_boundaries) and index_right < len(right_lane_boundaries):
            distance = np.linalg.norm(np.array(left_lane_boundaries[index_left]) - np.array(right_lane_boundaries[index_right]))
            if distance < max_distance:
                center_point = (np.array(left_lane_boundaries[index_left]) + np.array(right_lane_boundaries[index_right])) / 2
                centerline.append(center_point)
                index_left += 5
                index_right += 5
            else:
                index_left_new = index_left + 1
                index_right_new = index_right + 1
                if index_left_new >= len(left_lane_boundaries) or index_right_new >= len(right_lane_boundaries):
                    break
                distance_left = np.linalg.norm(np.array(left_lane_boundaries[index_left_new]) - np.array(right_lane_boundaries[index_right]))
                distance_right = np.linalg.norm(np.array(left_lane_boundaries[index_left]) - np.array(right_lane_boundaries[index_right_new]))
                if distance_left < distance_right:
                    index_left = index_left_new
                else:
                    index_right = index_right_new

        return centerline, []