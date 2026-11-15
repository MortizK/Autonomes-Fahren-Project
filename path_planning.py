import numpy as np


class PathPlanning:
    MAX_TRACK_WIDTH = 35.0

    def __init__(self):
        pass

    def create_centerline(
        self, longer_boundary: list, shorter_boundary: list
    ) -> list:
        centerline = []
        shorter_points = np.asarray(shorter_boundary, dtype=float)

        for point in longer_boundary:
            point = np.asarray(point, dtype=float)
            distances = np.linalg.norm(shorter_points - point, axis=1)
            valid_indices = np.flatnonzero(distances <= self.MAX_TRACK_WIDTH)
            if valid_indices.size == 0:
                continue

            nearest_index = valid_indices[np.argmin(distances[valid_indices])]
            midpoint = (point + shorter_points[nearest_index]) / 2.0
            centerline.append(midpoint.tolist())

        return centerline

    def calculate_curvature(self, line: list) -> list:
        """
        Calculate the curvature of a given line.

        args:
            line (list): A list of points representing the line.

        returns:
            curvature (list): The curvature of the line.
        """
        point_distance = 5

        if len(line) < 1 + 2 * point_distance:
            return []

        curvature = []
        for i in range(point_distance, len(line) - point_distance):
            p1 = np.array(line[i - point_distance])
            p2 = np.array(line[i])
            p3 = np.array(line[i + point_distance])

            # Calculate the radius of curvature using the three points
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)

            if a == 0 or b == 0 or c == 0:
                curvature.append(0)
                continue

            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            radius_of_curvature = (a * b * c) / (4 * area) if area != 0 else 0
            curvature.append(radius_of_curvature)

        return curvature

    def plan(self, left_lane_boundaries: list, right_lane_boundaries: list) -> tuple[list, list]:
        """
        Plan the trajectory based on the closest midpoint of both lane boundaries.
        """
        if len(left_lane_boundaries) == 0 or len(right_lane_boundaries) == 0:
            return [], []

        distance = np.linalg.norm(
            np.asarray(left_lane_boundaries[0])
            - np.asarray(right_lane_boundaries[0])
        )
        if distance < 15.0:
            return [], []

        centerline = self.create_centerline(
            right_lane_boundaries, left_lane_boundaries
        )
        curvature = self.calculate_curvature(centerline)
        return centerline, curvature

if __name__ == "__main__":
    # Example usage of the PathPlanning class
    path_planning = PathPlanning()

    # Example lane boundaries (straight)
    left_lane_boundaries = [[36, 77], [36, 76], [36, 75], [36, 74], [36, 73], [36, 72], [36, 71], [36, 70], [36, 69], [36, 68], [36, 67], [36, 66], [36, 65], [36, 64], [36, 63], [36, 62], [36, 61], [36, 60], [36, 59], [36, 58], [36, 57], [36, 56], [36, 55], [36, 54], [36, 53], [36, 52], [36, 51], [36, 50], [36, 49], [36, 48], [36, 47], [36, 46], [36, 45], [36, 44], [36, 43], [36, 42], [36, 41], [36, 40], [36, 39], [36, 38], [36, 37], [36, 36], [36, 35], [36, 34], [36, 33], [36, 32], [36, 31], [36, 30], [36, 29], [36, 28], [36, 27], [36, 26], [36, 25], [36, 24], [36, 23], [36, 22], [36, 21], [36, 20], [36, 19], [36, 18], [36, 17], [36, 16], [36, 15], [36, 14], [36, 13], [36, 12], [36, 11], [36, 10], [36, 9], [36, 8], [36, 7], [36, 6], [36, 5], [36, 4], [36, 3], [36, 2], [36, 1], [36, 0]]
    right_lane_boundaries = [[57, 77], [57, 76], [57, 75], [57, 74], [57, 73], [57, 72], [57, 71], [57, 70], [57, 69], [57, 68], [57, 67], [57, 66], [57, 65], [57, 64], [57, 63], [57, 62], [57, 61], [57, 60], [57, 59], [57, 58], [57, 57], [57, 56], [57, 55], [57, 54], [58, 53], [58, 52], [58, 51], [57, 50], [57, 49], [57, 48], [57, 47], [57, 46], [57, 45], [57, 44], [57, 43], [57, 42], [58, 41], [58, 40], [58, 39], [58, 38], [58, 37], [58, 36], [58, 35], [58, 34], [58, 33], [58, 32], [58, 31], [58, 30], [58, 29], [58, 28], [58, 27], [58, 26], [58, 25], [58, 24], [58, 23], [58, 22], [58, 21], [58, 20], [58, 19], [58, 18], [58, 17], [58, 16], [58, 15], [58, 14], [58, 13], [58, 12], [58, 11], [58, 10], [58, 9], [58, 8], [58, 7], [58, 6], [58, 5], [58, 4], [58, 3], [58, 2], [58, 1], [58, 0]]

    # Example lane boundaries (curved)
    left_lane_boundaries = [[36, 77], [36, 76], [36, 75], [36, 74], [36, 73], [36, 72], [36, 71], [36, 70], [36, 69], [36, 68], [36, 67], [36, 66], [36, 65], [36, 64], [36, 63], [36, 62], [36, 61], [36, 60], [36, 59], [36, 58], [36, 57], [36, 56], [36, 55], [36, 54], [36, 53], [36, 52], [36, 51], [36, 50], [36, 49], [37, 48], [36, 47], [36, 46], [36, 45], [36, 44], [35, 43], [34, 42], [33, 41], [32, 40], [31, 40], [30, 39], [29, 38], [28, 37], [27, 37], [26, 36], [25, 35], [24, 34], [23, 34], [22, 33], [21, 32], [20, 32], [19, 31], [18, 30], [17, 29], [16, 29], [15, 28], [14, 27], [13, 27], [12, 26], [11, 25], [10, 25], [9, 24], [8, 23], [7, 22], [6, 22], [5, 21], [4, 20], [3, 20], [2, 19], [1, 18], [0, 17]]
    right_lane_boundaries = [[58, 77], [58, 76], [58, 75], [58, 74], [58, 73], [58, 72], [58, 71], [58, 70], [58, 69], [58, 68], [58, 67], [58, 66], [58, 65], [58, 64], [58, 63], [58, 62], [58, 61], [58, 60], [58, 59], [58, 58], [58, 57], [58, 56], [58, 55], [58, 54], [58, 53], [58, 52], [58, 51], [58, 50], [58, 49], [58, 48], [58, 47], [58, 46], [58, 45], [58, 44], [58, 43], [57, 42], [57, 41], [57, 40], [57, 39], [56, 38], [56, 37], [56, 36], [56, 35], [55, 34], [55, 33], [55, 32], [54, 31], [53, 30], [53, 29], [52, 28], [52, 27], [51, 26], [51, 25], [50, 24], [49, 23], [48, 22], [47, 21], [46, 20], [45, 19], [44, 18], [43, 18], [42, 17], [41, 16], [40, 15], [39, 14], [38, 14], [37, 13], [36, 12], [35, 12], [34, 11], [33, 10], [32, 9], [31, 9], [30, 8], [29, 7], [28, 7], [27, 6], [26, 5], [25, 4], [24, 4], [23, 3], [22, 2], [21, 2], [20, 1], [19, 0]]

    centerline, curvature = path_planning.plan(left_lane_boundaries, right_lane_boundaries)

    print("Planned Centerline:", centerline)
    print("Curvature:", curvature)