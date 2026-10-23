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

        self.road_mask = self._create_road_mask(self.cut_image) == 0
        # self.debug_image = self.road_mask

        p = np.pad(self.road_mask, 1, mode='constant')

        neighbors = (
                          p[:-2, 1:-1] |
            p[1:-1, :-2]               | p[1:-1, 2:] |
                          p[2:, 1:-1]
        )

        self.road_mask = (self.road_mask & neighbors).astype(np.uint8) * 255
        # self.debug_image = self.road_mask

        # Phase 2: Lane Detection
        self.edge_mask = self._detect_edges(self.road_mask)
        # self.debug_image = self.edge_mask

        self.debug_image = observation.copy()
        for (y, x) in np.argwhere(self.edge_mask > 0):
            self.debug_image[y + 1, x + 1] = [255, 0, 0]

        # Phase 3: A* Path Planning - Find Start and Endpoints
        start_point = np.array([geo.CAR_Y_PX, geo.CAR_X_PX], dtype=np.int8)

        # Find all points on the border of the image (top, left, right)
        border_points = np.argwhere(self.edge_mask > 0)
        border_points = border_points[
            (border_points[:, 1] == 0) |                            # left border
            (border_points[:, 0] == 0) |                            # top border
            (border_points[:, 1] == self.edge_mask.shape[1] - 1)    # right border
        ]
        # Add points on the border of the image, which are part of the road mask
        for y in [0, self.edge_mask.shape[0] - 1]:
            for x in [0, self.edge_mask.shape[1] - 1]:
                if self.road_mask[y, x] > 0:
                    border_points = np.vstack((border_points, [y, x]))
        
        for y, x in border_points:
            self.debug_image[y + 1, x + 1] = [0, 255, 0]

        # Find Pairs of border points
        # Points which are direct neighbors are considered as one point
        clustered_border_points = []
        for i, (y, x) in enumerate(border_points):
            if i == 0:
                clustered_border_points.append((y, x))
                continue

            prev_y, prev_x = border_points[i - 1]
            if abs(y - prev_y) > 1 or abs(x - prev_x) > 1:
                clustered_border_points.append((y, x))
        print(f"Clustered: {len(clustered_border_points)}, Original: {len(border_points)}")

        mid_points = []
        for i in range(len(clustered_border_points)):
            y1, x1 = clustered_border_points[i]
            y2, x2 = clustered_border_points[(i + 1) % len(clustered_border_points)]
            mid_y = (y1 + y2) // 2
            mid_x = (x1 + x2) // 2
            if self.road_mask[mid_y +1, mid_x + 1] == 255 and self.edge_mask[mid_y, mid_x] == 0:
                mid_points.append((mid_y, mid_x))

        for y, x in mid_points:
            self.debug_image[y + 1, x + 1] = [0, 0, 255]

        return [], []
