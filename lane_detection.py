import numpy as np

import geometry as geo

class LaneDetection:
    debug_image: np.ndarray
    road_mask: np.ndarray
    edge_mask: np.ndarray

    def __init__(
        self, 
        threshold: list[int] = [145, 145, 145], 
        max_color_diff: int = 5
    ):
        self.debug_image = np.zeros((98, 98, 3), dtype=np.uint8)
        self.road_mask = np.zeros((84, 98), dtype=np.uint8)
        self.edge_mask = np.zeros((84, 98), dtype=np.uint8)
        self.frame = 0
        self.threshold = threshold
        self.max_color_diff = max_color_diff

    def _detectDomain(self, observation: np.ndarray):
        """
        Detects the color of the road in the given Domain
        """
        # Get the average color of the road in the bottom row of the image
        road_color = observation[geo.CAR_Y_PX-25:geo.CAR_Y_PX-15, geo.CAR_X_PX, :].mean(axis=0)
        self.threshold = road_color

    def _create_road_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        This function uses the color of the road to create a binary mask of the road.

        returns:
            road_mask (np.ndarray): A binary mask where road pixels are 255 and non-road pixels are 0.
        """
        red = frame[..., 0].astype(np.int16)
        green = frame[..., 1].astype(np.int16)
        blue = frame[..., 2].astype(np.int16)

        is_road = (abs(green - self.threshold[1]) < self.max_color_diff) & (abs(red - self.threshold[0]) < self.max_color_diff) & (abs(blue - self.threshold[2]) < self.max_color_diff)

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
            for dy in [-1, 0, 1]:   # First check the pixels upwards, then the pixels in the same row, then the pixels downwards
                for dx in [-1, 0, 1]:
                    new_x = last_x + dx
                    new_y = last_y + dy
                    if (
                        0 <= new_x < edge_mask.shape[1]
                        and 0 <= new_y < edge_mask.shape[0]
                        and edge_mask[new_y, new_x] > 0
                        and (new_x, new_y) not in lane_boundaries   # Avoid creating loops
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
        2. Else, find the closest white pixel to the left and right of the car.

        returns:
            left_start (tuple): The starting point for the left lane boundary.
            right_start (tuple): The starting point for the right lane boundary.
        """
        left_start = None
        right_start = None

        row = geo.CAR_Y_PX  # Starting Points just below the car, to avoid detecting the car as a lane boundary
        white_cols = np.where(edge_mask[row, :] > 0)[0]

        if len(white_cols) == 2:
            left_start = (white_cols[0], row)
            right_start = (white_cols[1], row)
        elif len(white_cols) > 2:
            # Find the closest white pixel to the left and right of the car
            on_the_left = white_cols[white_cols <= geo.CAR_X_PX]
            on_the_right = white_cols[white_cols > geo.CAR_X_PX]

            # Start with the closest white pixel to the left of the car for the left lane
            left_start = None if len(on_the_left) == 0 else (on_the_left[-1], row)

            # Start with the closest white pixel to the right of the car for the right lane
            right_start = None if len(on_the_right) == 0 else (on_the_right[0], row)

        return left_start, right_start

    def _track_lane(self, start: tuple[int, int], is_left_lane: bool) -> list[tuple[int, int]]:
        """
        Tracks the lane boundary starting from a given point and draw the lane boundary on the debug image.
        """
        lane_boundaries = self._follow_lane(start, self.edge_mask)

        # Draw the lane boundaries on the debug image
        for count, (x, y) in enumerate(lane_boundaries, start=1):
            intensity = 255 - count / len(lane_boundaries) * 100
            if is_left_lane:
                self.overlay_image[y + 1, x + 1] = [intensity, 0, 0]    # Red for left lane
            else:
                self.overlay_image[y + 1, x + 1] = [0, 0, intensity]    # Blue for right lane

        # White starting point
        self.overlay_image[start[1] + 1, start[0] + 1] = [255, 255, 255]

        return lane_boundaries

    def detect(self, observation: np.ndarray) -> tuple[list, list]:
        """
        Detects the lane boundaries in the given observation.

        returns:
            left_lane_boundaries (list): A list of points representing the left lane boundaries.
            right_lane_boundaries (list): A list of points representing the right lane boundaries.
        """
        self.overlay_image = observation.copy() # Used for debugging purposes, to visualize the detected lane boundaries

        if observation[-3, 12][0] == 0:
            self.frame = 0      # Reset the frame counter, when the car enters a new track, to detect the domain again
            self.threshold = observation[geo.CAR_Y_PX+1, geo.CAR_X_PX] # First Estimate of the road color

        if self.frame == 25:
            self._detectDomain(observation)
        self.frame += 1

        # Phase 1: Preprocessing
        self.cut_image = observation[0:84, 0:98].copy() # Cut the image
        self.cut_image[geo.CAR_Y_PX-6:geo.CAR_Y_PX+5, geo.CAR_X_PX-3:geo.CAR_X_PX+3] = self.threshold
        self.road_mask = self._create_road_mask(self.cut_image)
        # self.debug_image = self.road_mask.copy()

        # Phase 2: Lane Detection
        self.edge_mask = self._detect_edges(self.road_mask)
        # self.debug_image = self.edge_mask.copy()

        # Phase 3: Zuordnung zu den Fahrspuren
        left_lane_boundaries = []
        right_lane_boundaries = []

        left_start, right_start = self._get_starting_point(self.edge_mask)

        # Append the left and right lane boundaries from the detected edges
        if left_start is not None:
            left_lane_boundaries = self._track_lane(left_start, is_left_lane=True)

        if right_start is not None:
            right_lane_boundaries = self._track_lane(right_start, is_left_lane=False)

        self.debug_image = self.overlay_image

        return left_lane_boundaries, right_lane_boundaries
