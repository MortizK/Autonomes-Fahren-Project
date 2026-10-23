import numpy as np

import geometry as geo

class LaneDetection:
    debug_image: np.ndarray
    road_mask: np.ndarray
    edge_mask: np.ndarray
    pace_notes: str

    def __init__(self):
        self.debug_image = np.zeros((98, 98, 3), dtype=np.uint8)
        self.road_mask = np.zeros((84, 98), dtype=np.uint8)
        self.edge_mask = np.zeros((84, 98), dtype=np.uint8)

    def _create_road_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Finds pixels, which are to bright or to colorful, as these pixels must not be part of the road.

        This function is environment specific and may need to be adjusted if the environment changes. (for example if the environment has a different colored track)

        returns:
            road_mask (np.ndarray): A binary mask where road pixels are 255 and non-road pixels are 0.
        """
        LIMIT = 145

        red = frame[..., 0].astype(np.int16)
        green = frame[..., 1].astype(np.int16)
        blue = frame[..., 2].astype(np.int16)

        is_road = (green > LIMIT) | (red > LIMIT) | (blue > LIMIT)

        return is_road.astype(np.uint8) * 255

    def _detect_edges(self, road_mask: np.ndarray) -> np.ndarray:
        """
        Detects edges in the road mask using 2 x 1D convolutions.

        returns:
            edge_mask (np.ndarray): A binary mask where edge pixels are 255 and non-edge pixels are 0.
        """
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

        returns:
            left_lane_boundaries (list): A list of points representing the left lane boundaries.
            right_lane_boundaries (list): A list of points representing the right lane boundaries.
        """

        # Phase 1: Preprocessing
        self.cut_image = observation[0:84, 0:98]   # Cut the image

        # Maskierung des Autos
        self.cut_image[int(geo.CAR_Y_PX-5):int(geo.CAR_Y_PX+5), int(geo.CAR_X_PX-2):int(geo.CAR_X_PX+2)] = [100, 100, 100]

        self.road_mask = self._create_road_mask(self.cut_image)
        self.debug_image = self.road_mask

        # Phase 2: Lane Detection
        self.edge_mask = self._detect_edges(self.road_mask)
        # self.debug_image = self.edge_mask

        return [], []
