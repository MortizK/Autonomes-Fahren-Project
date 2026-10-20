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

        # Phase 1: Preprocessing
        # Ausschneiden der unteren 14 Pixel, da dort keine Fahrspuren sind
        self.debug_image = self.debug_image[0:84, 0:98]
        # Maskierung des Autos
        self.debug_image[66:80, 45:51] = [102, 102, 102]  # Schwarze Maske

        # Umwandlung in Graustufen
        # Track ist Weiß und hat Schwarze Ränder
        track_filter = [1.2, 1.2, 0]
        track_image = np.dot(self.debug_image[..., :3], track_filter)

        self.debug_image = track_image.astype(np.uint8)

        # Phase 2: Lane Detection
        

        # Phase 3: Zuordnung zu den Fahrspuren


        # (Phase 4: Funktionsanpassung, Mathematische Darstellung der Fahrspuren)
        pass
