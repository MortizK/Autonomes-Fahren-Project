import numpy as np


class LaneDetection:
    debug_image: np.ndarray
    road_mask: np.ndarray

    def __init__(self, green_tolerance: int = 30):
        self.debug_image = np.zeros((98, 98, 3), dtype=np.uint8)
        self.road_mask = np.zeros((84, 98), dtype=np.uint8)
        self.green_tolerance = green_tolerance

    def _create_road_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return 255 for road pixels and 0 for green pixels."""
        red = frame[..., 0].astype(np.int16)
        green = frame[..., 1].astype(np.int16)
        blue = frame[..., 2].astype(np.int16)

        is_green = (green > red + self.green_tolerance) & (
            green > blue + self.green_tolerance
        )
        return np.where(is_green, 0, 255).astype(np.uint8)

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
        self.debug_image[66:77, 45:51] = [100, 100, 100]  # Schwarze Maske

        # Alles, was nicht grün ist, wird als Straße markiert.
        self.road_mask = self._create_road_mask(self.debug_image)
        self.debug_image = self.road_mask

        # Phase 2: Lane Detection
        

        # Phase 3: Zuordnung zu den Fahrspuren


        # (Phase 4: Funktionsanpassung, Mathematische Darstellung der Fahrspuren)
        return [], []
