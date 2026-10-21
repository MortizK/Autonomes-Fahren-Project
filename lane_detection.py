import numpy as np


class LaneDetection:
    CAR_CENTER_X = 48
    CAR_BOTTOM_Y = 77
    debug_image: np.ndarray
    road_mask: np.ndarray
    edge_mask: np.ndarray
    first_image: np.ndarray

    def __init__(self, color_tolerance: int = 100):
        self.debug_image = np.zeros((98, 98, 3), dtype=np.uint8)
        self.road_mask = np.zeros((84, 98), dtype=np.uint8)
        self.edge_mask = np.zeros((84, 98), dtype=np.uint8)
        self.first_image = None
        self.color_tolerance = color_tolerance

    def _create_road_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Finds pixels, which are to bright or to colorful, as these pixels must not be part of the road.

        This function is environment specific and may need to be adjusted if the environment changes. (for example if the environment has a different colored track)

        returns:
            road_mask (np.ndarray): A binary mask where road pixels are 255 and non-road pixels are 0.
        """
        LIMIT = 120

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

    def _follow_lane(self, start_point: tuple[int, int], edge_mask: np.ndarray) -> list[tuple[int, int]]:
        """
        Follows the lane from a starting point.

        perquisite:
            The starting point must be a white pixel in the edge mask.

        returns:
            lane_boundaries (list): A list of points representing the lane boundaries.
        """
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
                    continue    # if we did not find a new point, continue searching
                break   # If we did find a new point, we can start searching for the next point from there
            else:
                break   # If we did not find a new point, we are done
        return lane_boundaries

    def _get_starting_point(self, edge_mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Finds the starting points for the left and right lane boundaries. In multiple cases:

        1. If there are only two white pixels, they must be the starting points for the left and right lane boundaries.
        2. If the car is on the road, find the closest white pixel to the left and right of the car.
        3. If the car is not on the road, return None for both starting points (Currently, we do not handle this case, as we assume the car is always on the road).

        returns:
            left_start (tuple): The starting point for the left lane boundary.
            right_start (tuple): The starting point for the right lane boundary.
        """
        left_start = None
        right_start = None

        white_cols = np.where(edge_mask[77, :] > 0)[0]  # Get the x-coordinates of the white pixels in the bottom row

        if len(white_cols) == 2:
            left_start = (white_cols[0], self.CAR_BOTTOM_Y)
            right_start = (white_cols[1], self.CAR_BOTTOM_Y)
        elif self.road_mask[self.CAR_BOTTOM_Y, self.CAR_CENTER_X] == 0:
            # If the car is on the road, find the closest white pixel to the left and right of the car
            on_the_left = white_cols[white_cols < self.CAR_CENTER_X]
            on_the_right = white_cols[white_cols > self.CAR_CENTER_X]

            # Start with the closest white pixel to the left of the car for the left lane
            left_start = None if len(on_the_left) == 0 else (on_the_left[-1], self.CAR_BOTTOM_Y)

            # Start with the closest white pixel to the right of the car for the right lane
            right_start = None if len(on_the_right) == 0 else (on_the_right[0], self.CAR_BOTTOM_Y)
        else:
            # TODO: Do we need behavior for when the car is not on the road? For now, we will just return empty lists.
            left_start = None
            right_start = None

        return left_start, right_start

    def detect(self, observation: np.ndarray) -> tuple[list, list]:
        """
        Detects the lane boundaries in the given observation.

        returns:
            left_lane_boundaries (list): A list of points representing the left lane boundaries.
            right_lane_boundaries (list): A list of points representing the right lane boundaries.
        """

        self.debug_image = observation.copy()

        if self.first_image is None:
            self.first_image = observation.copy()

        # Phase 1: Preprocessing
        self.cut_image = self.debug_image[0:84, 0:98]   # Cut the image

        # Maskierung des Autos
        self.cut_image[66:self.CAR_BOTTOM_Y, self.CAR_CENTER_X-2:self.CAR_CENTER_X+2] = [100, 100, 100]

        self.road_mask = self._create_road_mask(self.cut_image)

        # Phase 2: Lane Detection
        self.edge_mask = self._detect_edges(self.road_mask)
        # Runtime: until here p95 6ms

        # Phase 3: Zuordnung zu den Fahrspuren
        left_lane_boundaries = []
        right_lane_boundaries = []

        left_start, right_start = self._get_starting_point(self.edge_mask)

        # Append the left and right lane boundaries from the detected edges
        if left_start is not None:
            left_lane_boundaries = self._follow_lane(left_start, self.edge_mask)

            # draw the left lane boundaries on the debug image
            for count, (x, y) in enumerate(left_lane_boundaries, start=1):
                self.debug_image[y + 1, x + 1] = [255 - count / len(left_lane_boundaries) * 100, 0, 0]   # Red for left lane
            self.debug_image[left_start[1] + 1, left_start[0] + 1] = [255, 255, 255]
        
        if right_start is not None:
            right_lane_boundaries = self._follow_lane(right_start, self.edge_mask)

            for count, (x, y) in enumerate(right_lane_boundaries, start=1):
                self.debug_image[y + 1, x + 1] = [0, 0, 255 - count / len(right_lane_boundaries) * 100]   # Blue for right lane
            self.debug_image[right_start[1] + 1, right_start[0] + 1] = [255, 255, 255]
        # Runtime: until here p95 14ms

        # (Phase 4: Funktionsanpassung, Mathematische Darstellung der Fahrspuren)

        return left_lane_boundaries, right_lane_boundaries
