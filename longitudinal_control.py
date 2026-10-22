import numpy as np

import geometry as geo

# Measured on asphalt [units/s^2]
GAS_ACCEL = 44.0  # full gas
BRAKE_ACCEL = 300.0  # per brake value, linear up to ~0.7
GRIP_ACCEL = 219.0  # tire friction limit


class LongitudinalControl:
    """
    Target speed from curvature and braking distance, tracked by a PID controller.

    Corner speed v = sqrt(a_lat / kappa), a corner at distance s allows sqrt(v^2 + 2 * a_brake * s).
    The PID output is a desired acceleration, mapped to gas or brake.
    """

    def __init__(
        self,
        kp: float = 8.0,
        ki: float = 1.0,
        kd: float = 0.05,
        max_speed: float = 100.0,
        min_speed: float = 30.0,
        lateral_accel: float = 240.0,
        brake_accel: float = 150.0,
        curvature_window: int = 8,
        max_brake: float = 0.7,
    ):
        self.kp = kp  # [1/s]
        self.ki = ki  # [1/s^2]
        self.kd = kd  # on the measured acceleration

        self.max_speed = max_speed
        self.min_speed = min_speed
        self.lateral_accel = lateral_accel  # on the centerline, above the grip limit since curves are cut
        self.brake_accel = brake_accel  # planned deceleration before curves
        self.curvature_window = curvature_window  # points per side, short curves look flatter
        self.max_brake = max_brake  # wheels lock above ~0.7

        self.reset()

    def reset(self) -> None:
        """Resets the controller state."""
        self._frame = 0
        self._target_speed = self.min_speed
        self._integral = 0.0
        self._last_speed = None

    def predict_target_speed(self, trajectory: list) -> float:
        """
        Predicts the target speed based on the curvature of the trajectory ahead.

        Args:
            trajectory (list): A list of points representing the planned trajectory.

        Returns:
            float: The predicted target speed.
        """
        zoom = geo.camera_zoom(self._frame)
        self._frame += 1

        # Path ignored while zooming in (pixels too coarse)
        path = None
        if len(trajectory) >= 2 and zoom >= geo.ZOOM / 2:
            path = geo.path_ahead(geo.to_vehicle_frame(trajectory, zoom), np.zeros(2))

        # No path: slow down with the planned deceleration
        if path is None or len(path) <= 2 * self.curvature_window:
            self._target_speed = max(self._target_speed - self.brake_accel * geo.DT, self.min_speed)
            return self._target_speed

        # Corner speed at every path point
        curvature = np.abs(geo.curvature(path, self.curvature_window))
        corner_speed = np.sqrt(self.lateral_accel / np.maximum(curvature, 1e-6))

        # Max speed that can still brake to each corner speed
        distance = np.arange(curvature.size) * geo.PATH_SPACING
        reachable = np.sqrt(corner_speed**2 + 2 * self.brake_accel * distance)

        # A sharp turn may follow right after the visible path
        horizon = np.sqrt(self.min_speed**2 + 2 * self.brake_accel * (len(path) - 1) * geo.PATH_SPACING)

        self._target_speed = float(np.clip(min(reachable.min(), horizon), self.min_speed, self.max_speed))
        return self._target_speed

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
        error = target_speed - current_speed

        # D-term on the measurement, no kick when the target jumps
        if self._last_speed is None:
            self._last_speed = current_speed
        accel = (current_speed - self._last_speed) / geo.DT
        self._last_speed = current_speed

        demand = self.kp * error + self.ki * self._integral - self.kd * accel

        # Anti-windup: integrate only while not saturated
        saturated = demand > GAS_ACCEL or demand < -self.max_brake * BRAKE_ACCEL
        if not saturated or np.sign(error) != np.sign(demand):
            self._integral += error * geo.DT

        # Friction circle: gas gets only the grip left after cornering
        lateral = current_speed**2 * np.tan(min(abs(steering_angle), geo.MAX_STEER)) / geo.WHEELBASE
        grip = np.sqrt(max(1.0 - (lateral / GRIP_ACCEL) ** 2, 0.0))

        # Braking stays allowed in curves
        if demand >= 0.0:
            throttle = min(demand / GAS_ACCEL, grip)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-demand / BRAKE_ACCEL, self.max_brake)

        return float(throttle), float(brake)
