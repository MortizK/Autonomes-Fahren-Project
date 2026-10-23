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
        return 0.0
