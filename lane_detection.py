import numpy as np


class LaneDetection:
    debug_image: np.ndarray
    road_mask: np.ndarray

    def __init__(self, color_tolerance: int = 30):
        self.debug_image = np.zeros((98, 98, 3), dtype=np.uint8)
        self.road_mask = np.zeros((84, 98), dtype=np.uint8)
        self.color_tolerance = color_tolerance

    def _create_road_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return 255 for road pixels and 0 for green pixels."""
        red = frame[..., 0].astype(np.int16)
        green = frame[..., 1].astype(np.int16)
        blue = frame[..., 2].astype(np.int16)

        is_road = (green > 120) | (red > 120) | (blue > 120)

        return is_road.astype(np.uint8) * 255

    def _detect_edges(self, road_mask: np.ndarray) -> np.ndarray:
        """Detect edges in the road mask."""
        self.road_mask = road_mask
        kernel_x = np.array([-1, 2, -1], dtype=np.int8)
        edges_x = np.convolve(self.road_mask.flatten(), kernel_x, mode='same').reshape(self.road_mask.shape)

        road_mask_rotated = np.rot90(self.road_mask, k=1)
        kernel_y = np.array([-1, 2, -1], dtype=np.int8).reshape(-1, 1)
        edges_y = np.convolve(road_mask_rotated.flatten(), kernel_y.flatten(), mode='same').reshape(road_mask_rotated.shape)
        edges_y = np.rot90(edges_y, k=-1)

        # Remove the Edges at the borders to avoid false positives
        edges_x = edges_x[1:-1, 1:-1]
        edges_y = edges_y[1:-1, 1:-1]

        return edges_x + edges_y

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
        self.debug_image[66:77, 45:51] = [100, 100, 100]

        self.road_mask = self._create_road_mask(self.debug_image)
        self.debug_image = self.road_mask

        # Phase 2: Lane Detection
        self.edge_mask = self._detect_edges(self.road_mask)
        self.debug_image = self.edge_mask

        # Phase 3: Zuordnung zu den Fahrspuren


        # (Phase 4: Funktionsanpassung, Mathematische Darstellung der Fahrspuren)
        return [], []
