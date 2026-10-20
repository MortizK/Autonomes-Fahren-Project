import numpy as np


class LaneDetection:
    debug_image: np.ndarray

    def __init__(self):
        self.debug_image = np.zeros((98, 98, 3), dtype=np.uint8)
        pass

    def detect(self, observation: np.ndarray) -> tuple[list, list]:
        """
        Detects the lane boundaries in the given observation.

        args:
            observation (np.ndarray): 98 x 98 rgb image.

        returns:
            left_lane_boundaries (list): A list of points representing the left lane boundaries.
            right_lane_boundaries (list): A list of points representing the right lane boundaries.
        """
        self.debug_image = observation.copy()
        pass
