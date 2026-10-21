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
    pace_notes: str

    def __init__(self):
        self.debug_image = np.zeros((98, 98, 3), dtype=np.uint8)
        self.road_mask = np.zeros((84, 98), dtype=np.uint8)
        self.edge_mask = np.zeros((84, 98), dtype=np.uint8)
        self.first_image = None
        self.first_image_road_mask = None
        self.pace_notes = ""

    def _create_road_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return 255 for road pixels and 0 for green pixels."""
        LIMIT = 135

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

    def _get_starting_point(self, edge_mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
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

    def _order_closed_contour(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.uint8)  # Shape: (N, 2)

        center = points.mean(axis=0)

        # Mirror the points across the center
        mirror_points = 2 * center - points

        angles = np.arctan2(
            mirror_points[:, 0] - center[0],   # dx
            mirror_points[:, 1] - center[1]    # dy
        )

        self.debug_image[int(center[0]), int(center[1])] = 0

        distance_list = np.linalg.norm(points - center, axis=1)

        return points[np.argsort(angles)], distance_list[np.argsort(angles)]

    def _create_pace_notes(self, road_mask: np.ndarray) -> str:
            """
            Create pace notes based on the road mask.
    
            returns:
                pace_notes (str): A string representing the pace notes.
            """
            road_mask = road_mask[57:84, 17:51]  # Cut the image to the relevant area
            np.set_printoptions(threshold=np.inf)
            self.debug_image = road_mask.copy()  # For debugging purposes
    
            point_list = np.argwhere(road_mask == 0)  # Get the coordinates of the road pixels
            
            print(f"total points: {len(point_list)}")
            if len(point_list) == 0:
                return "MC1 MC2BC"
    
            # Phase 0: Sort the points by angle around the track center
            ordered_point_list, distance_list = self._order_closed_contour(point_list)

            n = len(ordered_point_list)
            averaged_vector_list = []

            # Phase 1: Average the directions toward the 4 closest of the next 10 points
            window_size = min(10, n - 1)
            choose = min(4, window_size)

            for i in range(n):
                x, y = ordered_point_list[i]

                # Get the next 10 points along the closed contour
                candidate_indices = [
                    (i + j + 1) % n
                    for j in range(window_size)
                ]

                candidates = np.asarray(
                    [ordered_point_list[j] for j in candidate_indices],
                    dtype=float
                )

                # Calculate displacement vectors from the current point
                displacements = candidates - np.array([x, y])

                # Calculate Euclidean distances
                distances = np.linalg.norm(displacements, axis=1)

                # Select the 4 spatially closest candidates
                closest_indices = np.argsort(distances)[:choose]

                # Average their displacement vectors
                avg_vector = np.mean(displacements[closest_indices], axis=0)

                averaged_vector_list.append(tuple(avg_vector))

            # Phase 0/3 - Calculate the angles between the vectors to find the corners
            angle_list = []
            for i in range(len(averaged_vector_list)):
                x1, y1 = averaged_vector_list[i]
                x2, y2 = averaged_vector_list[(i + 1) % len(averaged_vector_list)]
                angle = np.arctan2(y2, x2) - np.arctan2(y1, x1)
                angle_list.append(angle)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(self.debug_image, cmap='gray')
            plt.title('Road Mask with Ordered Points')
            plt.subplot(1, 2, 2)
            plt.plot(angle_list)
            plt.title('Angle List')
            plt.savefig("debug_angle_list.png")
            plt.show()
            plt.close()
    
            # Phase 1 - Create the line segments from the points
            line_segments = []
    
            # Phase 2 - Create the Pace Notes from the points
            pace_notes = ["MC1"] # Start with a default pace note
    
            pace_notes.append("MC2BC") # End with a default pace note
            return " ".join(pace_notes)

    def detect(self, observation: np.ndarray) -> tuple[list, list]:
        """
        Detects the lane boundaries in the given observation.

        args:
            observation (np.ndarray): 98 x 98 rgb image.

        returns:
            left_lane_boundaries (list): A list of points representing the left lane boundaries.
            right_lane_boundaries (list): A list of points representing the right lane boundaries.
        """

        if self.first_image is None:
            self.first_image = observation.copy()

        # Phase 1: Preprocessing
        # Ausschneiden der unteren 14 Pixel, da dort keine Fahrspuren sind
        self.cut_image = observation[0:84, 0:98]
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
            left_lane_boundaries = self._follow_lane(left_start, self.edge_mask)
            # draw the left lane boundaries on the debug image
            for count, (x, y) in enumerate(left_lane_boundaries, start=1):
                self.overlay_image[y + 1, x + 1] = [255 - count / len(left_lane_boundaries) * 100, 0, 0]   # Red for left lane
            self.overlay_image[left_start[1] + 1, left_start[0] + 1] = [255, 255, 255]
        
        if right_start is not None:
            right_lane_boundaries = self._follow_lane(right_start, self.edge_mask)
            for count, (x, y) in enumerate(right_lane_boundaries, start=1):
                self.overlay_image[y + 1, x + 1] = [0, 0, 255 - count / len(right_lane_boundaries) * 100]   # Blue for right lane
            self.overlay_image[right_start[1] + 1, right_start[0] + 1] = [255, 255, 255]
        # Runtime: until here p95 14ms

        # self.debug_image = self.overlay_image
        # self.debug_image = self.first_image

        # (Phase 4: Funktionsanpassung, Mathematische Darstellung der Fahrspuren)
        return left_lane_boundaries, right_lane_boundaries
