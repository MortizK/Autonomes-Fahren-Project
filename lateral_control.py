import numpy as np


class LateralControl:

    def __init__(self):
        self._car_position = np.array([48, 64])

    def control(self, trajectory: list, speed: float) -> float:
        """
        Controls the lateral movement of the car based on the planned trajectory and current speed.

        Args:
            trajectory (list): A list of points representing the planned trajectory.
            speed (float): The current speed of the car.

        Returns:
            steering_angle (float): -1 is full left, +1 is full right.
        """
        if len(trajectory) == 0:
            return 0.0

        # TODO: only works till speed of ~45
        # Phase 1: Get the target point
        lookahead_points = 8
        target_point = trajectory[-1]
        if len(trajectory) > lookahead_points:
            target_point = trajectory[lookahead_points]
        print(f"Target point: {target_point}, Speed: {speed}")

        # Phase 2: Calculate the steering angle based on the target point
        center_pixel = np.array([48, 64])

        # Map the distance between -1 and 1 with an activation function
        angle = np.clip((target_point[0] - center_pixel[0]) / 48, -1, 1)
        # print(f"Steering angle: {angle}")

        return angle

