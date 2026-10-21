import numpy as np


class LongitudinalControl:
    def __init__(self):
        pass

    def control(self, current_speed: float, target_speed: float, steering_angle: float) -> tuple[float, float]:
        """
        Controls the longitudinal movement of the car based on the current speed, target speed, and steering angle.

        Args:
            current_speed (float): The current speed of the car.
            target_speed (float): The desired speed of the car.
            steering_angle (float): The current steering angle of the car.

        Returns:
            throttle (float): 0 to 1.
            brake (float): 0 to 1.
        """
        return 0.0, 0.0

    def predict_target_speed(self, curvature: float) -> float:
        """
        Predicts the target speed based on the current state of the car.

        Returns:
            float: The predicted target speed.
        """
        return 0.0
