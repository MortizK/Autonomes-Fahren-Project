import numpy as np


class LaneDetection:
    debug_image: np.ndarray
    road_mask: np.ndarray
    edge_mask: np.ndarray

    def __init__(self, color_tolerance: int = 100):
        self.debug_image = np.zeros((98, 98, 3), dtype=np.uint8)
        self.road_mask = np.zeros((84, 98), dtype=np.uint8)
        self.edge_mask = np.zeros((84, 98), dtype=np.uint8)
        self.color_tolerance = color_tolerance

    def _create_road_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return 255 for road pixels and 0 for green pixels."""
        LIMIT = 120

        red = frame[..., 0].astype(np.int16)
        green = frame[..., 1].astype(np.int16)
        blue = frame[..., 2].astype(np.int16)

        is_road = (green > LIMIT) | (red > LIMIT) | (blue > LIMIT)

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

    def _follow_lane(self, start_point: tuple[int, int], edge_mask: np.ndarray) -> list[tuple[int, int]]:
        lane_boundaries = [start_point]
        last_x, last_y = start_point
        while True:
            # Find the next point on the left lane boundary
            # The lane may do a 180 Turn, so we need to check the surrounding pixels
            last_x, last_y = lane_boundaries[-1]
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    new_x = last_x + dx
                    new_y = last_y + dy
                    if (
                        0 <= new_x < edge_mask.shape[1]
                        and 0 <= new_y < edge_mask.shape[0]
                        and edge_mask[new_y, new_x] > 0
                        and (new_x, new_y) not in lane_boundaries
                    ):
                        lane_boundaries.append((new_x, new_y))
                        break
                    else:
                        continue
                else:
                    continue
                break
            else:
                break
        return lane_boundaries

    def detect(self, observation: np.ndarray) -> tuple[list, list]:
        """
        Detects the lane boundaries in the given observation.

        args:
            observation (np.ndarray): 98 x 98 rgb image.

        returns:
            left_lane_boundaries (list): A list of points representing the left lane boundaries.
            right_lane_boundaries (list): A list of points representing the right lane boundaries.
        """
        CAR_CENTER_X = 48
        CAR_BOTTOM_Y = 77

        self.debug_image = observation.copy()

        # Phase 1: Preprocessing
        # Ausschneiden der unteren 14 Pixel, da dort keine Fahrspuren sind
        self.debug_image = self.debug_image[0:84, 0:98]
        # Maskierung des Autos
        self.debug_image[66:CAR_BOTTOM_Y, CAR_CENTER_X-2:CAR_CENTER_X+2] = [100, 100, 100]

        self.road_mask = self._create_road_mask(self.debug_image)
        # self.debug_image = self.road_mask

        # Phase 2: Lane Detection
        self.edge_mask = self._detect_edges(self.road_mask)
        # self.debug_image = self.edge_mask

        # Phase 3: Zuordnung zu den Fahrspuren
        left_lane_boundaries = []
        right_lane_boundaries = []

        # Find the Start from the side of the car
        white_cols = np.where(self.edge_mask[CAR_BOTTOM_Y] > 0)[0]

        if len(white_cols) == 0:
            return left_lane_boundaries, right_lane_boundaries
        else:
            on_the_left = white_cols[white_cols < CAR_CENTER_X]
            on_the_right = white_cols[white_cols > CAR_CENTER_X]
            # Start with the closest white pixel to the left of the car for the left lane
            left_start = (on_the_left[-1] if len(on_the_left) > 0 else white_cols[0], CAR_BOTTOM_Y)
            # Start with the closest white pixel to the right of the car for the right lane
            right_start = (on_the_right[0] if len(on_the_right) > 0 else white_cols[-1], CAR_BOTTOM_Y)

        # Append the left and right lane boundaries from the detected edges
        left_lane_boundaries = self._follow_lane(left_start, self.edge_mask)
        right_lane_boundaries = self._follow_lane(right_start, self.edge_mask)

        overlay_image = observation.copy()

        # overlay the detected lane boundaries on the debug image
        count = 0
        for x, y in left_lane_boundaries:
            # Offset due to smaller debug image size
            count += 1
            overlay_image[y + 1, x + 1] = [255 - count * 2, 0, 0]  # Red for left lane

        count = 0
        for x, y in right_lane_boundaries:
            # Offset due to smaller debug image size
            count += 1
            overlay_image[y + 1, x + 1] = [0, 0, 255 - count * 2]  # Blue for right lane

        self.debug_image = overlay_image

        for i in white_cols:
            self.debug_image[CAR_BOTTOM_Y + 1, i + 1] = [255, 255, 255]  # White for detected edges

        # (Phase 4: Funktionsanpassung, Mathematische Darstellung der Fahrspuren)
        return left_lane_boundaries, right_lane_boundaries
