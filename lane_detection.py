import cv2
import numpy as np

import matplotlib.pyplot as plt

class LaneDetection:
    CAR_CENTER_X = 48
    CAR_BOTTOM_Y = 77
    debug_image: np.ndarray
    road_mask: np.ndarray
    edge_mask: np.ndarray
    first_image: np.ndarray
    first_image_road_mask: np.ndarray
    ordered_points: np.ndarray
    pace_notes: str

    def __init__(self):
        self.debug_image = np.zeros((98, 98, 3), dtype=np.uint8)
        self.road_mask = np.zeros((84, 98), dtype=np.uint8)
        self.edge_mask = np.zeros((84, 98), dtype=np.uint8)
        self.first_image = None
        self.first_image_road_mask = None
        self.ordered_points = None
        self.pace_notes = ""

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

    def _get_starting_point(self, edge_mask: np.ndarray, row=77) -> tuple[tuple[int, int], tuple[int, int]]:
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

        white_cols = np.where(edge_mask[row, :] > 0)[0]  # Get the x-coordinates of the white pixels in the bottom row

        if len(white_cols) == 2:
            left_start = (white_cols[0], row)
            right_start = (white_cols[1], row)
        elif self.road_mask[row, self.CAR_CENTER_X] == 0:
            # If the car is on the road, find the closest white pixel to the left and right of the car
            on_the_left = white_cols[white_cols < self.CAR_CENTER_X]
            on_the_right = white_cols[white_cols > self.CAR_CENTER_X]

            # Start with the closest white pixel to the left of the car for the left lane
            left_start = None if len(on_the_left) == 0 else (on_the_left[-1], row)

            # Start with the closest white pixel to the right of the car for the right lane
            right_start = None if len(on_the_right) == 0 else (on_the_right[0], row)
        else:
            # TODO: Do we need behavior for when the car is not on the road? For now, we will just return empty lists.
            left_start = None
            right_start = None

        return left_start, right_start

    def _nearest_neighbor_walk(self, point_list: np.ndarray, start_point: np.ndarray, previous_direction=np.array([-1.0, 0.0])) -> tuple[list[np.ndarray], bool]:
        """
        Perform a nearest neighbor walk through the given points.

        args:
            point_list (np.ndarray): An array of points to walk through.

        returns:
            walk_points (list[np.ndarray]): A list of points representing the walk.
            closed_loop (bool): True if the walk forms a closed loop, False otherwise.
        """
        current_point = np.asarray(start_point, dtype=float)

        walk_points = [current_point.copy()]
        remaining_points = np.asarray(point_list[1:], dtype=float)

        max_distance = 6.0  # maximum distance to consider
        alpha = 0.7         # weight for distance score
        beta = 0.3          # weight for turn score

        while len(remaining_points) > 0:
            displacements = remaining_points - current_point
            distances = np.linalg.norm(displacements, axis=1)

            # Exclude points that are too far away
            valid = (distances > 0) & (distances <= max_distance)

            # If no valid points are found and loop was not closed, 
            # return empty list to indicate failure to create a closed loop
            if not np.any(valid):
                return walk_points, False

            candidate_indices = np.flatnonzero(valid)
            candidates = remaining_points[valid]
            candidate_distances = distances[valid]

            directions = displacements[valid] / candidate_distances[:, None]

            # Directional alignment: 1 means straight ahead, -1 means backwards
            alignment = directions @ previous_direction

            # Penalize sharp turns
            distance_score = candidate_distances / max_distance
            turn_score = 1.0 - alignment

            scores = alpha * distance_score + beta * turn_score

            best_local_index = np.argmin(scores)
            best_index = candidate_indices[best_local_index]

            next_point = remaining_points[best_index]
            next_direction = directions[best_local_index]

            # check for loop closure: 
            closure_distance = 2.0
            distances_to_start = np.linalg.norm(next_point - start_point)
            if len(walk_points) > 4 and distances_to_start < closure_distance:
                return walk_points, True

            walk_points.append(next_point.copy())

            current_point = next_point
            previous_direction = next_direction

            # Remove all points which are closer but where not chosen, to avoid going back on the track
            remove_mask = (distances <= candidate_distances[best_local_index]) & (valid)
            remaining_points = np.delete(remaining_points, np.flatnonzero(remove_mask), axis=0)
        return walk_points, False

    def _create_pace_notes(self, road_mask: np.ndarray) -> str:
            """
            Create pace notes based on the road mask.
    
            returns:
                pace_notes (str): A string representing the pace notes.
            """
            # road_mask = road_mask[35:84, 0:52]  # Cut the image to the relevant area
            # road_mask = road_mask[57:84, 17:51]  # Cut the image to the relevant area
            np.set_printoptions(threshold=np.inf)
            # self.debug_image = road_mask.copy()  # For debugging purposes
    
            point_list = np.argwhere(road_mask == 0)  # Get the coordinates of the road pixels
            
            if len(point_list) == 0:
                return "No Road Found, your on your own!"
    
            # Phase 1 - Nearest Neighbor Walk (with directionality)
            start_point = [self.CAR_BOTTOM_Y - 5, self.CAR_CENTER_X]
            walk_points, closed_loop = self._nearest_neighbor_walk(point_list, start_point=start_point, previous_direction=np.array([-1.0, 0.0]))

            if not closed_loop:
                return "No Track Data, your on your own!"

            self.ordered_points = np.zeros(road_mask.shape, dtype=np.uint8)
            for i in range(len(walk_points) - 1):
                x1, y1 = walk_points[i]
                self.ordered_points[int(x1), int(y1)] = 50 + i
            # print(self.ordered_points)
    
            # Phase N - Create the Pace Notes from the points
            pace_notes = ["MC1"] # Start with a default pace note
    
            pace_notes.append("MC2BC") # End with a default pace note
            return " ".join(pace_notes)

    def _track_left_lane(self, left_start: tuple[int, int]) -> list[tuple[int, int]]:
            left_lane_boundaries = self._follow_lane(left_start, self.edge_mask)

            # draw the left lane boundaries on the debug image
            for count, (x, y) in enumerate(left_lane_boundaries, start=1):
                self.overlay_image[y + 1, x + 1] = [255 - count / len(left_lane_boundaries) * 100, 0, 0]   # Red for left lane
            self.overlay_image[left_start[1] + 1, left_start[0] + 1] = [255, 255, 255]
            return left_lane_boundaries

    def _track_right_lane(self, right_start: tuple[int, int]) -> list[tuple[int, int]]:
            right_lane_boundaries = self._follow_lane(right_start, self.edge_mask)

            # draw the right lane boundaries on the debug image
            for count, (x, y) in enumerate(right_lane_boundaries, start=1):
                self.overlay_image[y + 1, x + 1] = [0, 0, 255 - count / len(right_lane_boundaries) * 100]   # Blue for right lane
            self.overlay_image[right_start[1] + 1, right_start[0] + 1] = [255, 255, 255]
            return right_lane_boundaries

    def detect(self, observation: np.ndarray) -> tuple[list, list]:
        """
        Detects the lane boundaries in the given observation.

        returns:
            left_lane_boundaries (list): A list of points representing the left lane boundaries.
            right_lane_boundaries (list): A list of points representing the right lane boundaries.
        """

        if self.first_image is None:
            self.first_image = observation.copy()

        # Phase 1: Preprocessing
        self.cut_image = observation[0:84, 0:98]   # Cut the image

        # Maskierung des Autos
        if self.first_image_road_mask is not None:
            self.cut_image[66:self.CAR_BOTTOM_Y, self.CAR_CENTER_X-2:self.CAR_CENTER_X+2] = [100, 100, 100]

        self.road_mask = self._create_road_mask(self.cut_image)
        # self.debug_image = self.road_mask

        if self.first_image_road_mask is None:
            self.first_image_road_mask = self.road_mask
            self.pace_notes = self._create_pace_notes(self.first_image_road_mask)
            print(f"Co-Driver: Pace Notes '{self.pace_notes}'")

        # Phase 2: Lane Detection
        self.edge_mask = self._detect_edges(self.road_mask)
        # self.debug_image = self.edge_mask

        # Phase 3: Zuordnung zu den Fahrspuren
        self.overlay_image = observation.copy()
        
        left_lane_boundaries = []
        right_lane_boundaries = []
        # Runtime: until here p95 6ms

        left_start, right_start = self._get_starting_point(self.edge_mask)

        # Append the left and right lane boundaries from the detected edges
        if left_start is not None:
            left_lane_boundaries = self._track_left_lane(left_start)
        
        if right_start is not None:
            right_lane_boundaries = self._track_right_lane(right_start)
        # Runtime: until here p95 14ms

        # Phase 4: Lookahead if curve is cut of from the image
        # last_left = left_lane_boundaries[-1] if left_lane_boundaries else (0, 0)
        # last_right = right_lane_boundaries[-1] if right_lane_boundaries else (0, 0)
        # print(f"Last Left: {last_left}, Last Right: {last_right}")
        # if last_left[1] > last_right[1] and last_right[1] == 0:
        #     # right is cut of on the top
        #     left_start, right_start = self._get_starting_point(self.edge_mask, row=last_right[1])
        #     print(f"New Start Points: Left: {left_start}, Right: {right_start}")
        #     if left_start is not None:
        #         right_lane_boundaries += self._track_right_lane(left_start)

        # self.debug_image = self.overlay_image
        # self.debug_image = self.first_image
        self.debug_image = self.ordered_points

        # (Phase 5: Funktionsanpassung, Mathematische Darstellung der Fahrspuren)

        return left_lane_boundaries, right_lane_boundaries
