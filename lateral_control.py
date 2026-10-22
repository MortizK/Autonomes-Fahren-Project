import numpy as np

import geometry as geo

HOLD_DECAY = 0.9  # steering decay per frame without a path


class LateralControl:
    """
    Stanley controller at the front axle:

        delta = psi + arctan(k * e / (k_soft + v)) + k_d * dpsi/dt
    """

    def __init__(self, gain: float = 1.0, softening: float = 1.0, damping: float = 0.02, preview: float = 12.0):
        self.gain = gain  # k [1/s], cross track error weight
        self.softening = softening  # k_soft [units/s], limits the gain at low speed
        self.damping = damping  # k_d [s], D-term on the heading error
        self.preview = preview  # chord length [units] for the heading error, classic Stanley ~1
        self.reset()

    def reset(self) -> None:
        """Resets the controller state."""
        self._frame = 0
        self._steering = 0.0
        self._heading_error = None

    def control(self, trajectory: list, speed: float) -> float:
        """
        Controls the lateral movement of the car based on the planned trajectory and current speed.

        Args:
            trajectory (list): A list of points representing the planned trajectory.
            speed (float): The current speed of the car.

        Returns:
            steering_angle (float): Target wheel angle in rad, positive = right, lock at +/-0.4.
        """
        zoom = geo.camera_zoom(self._frame)
        self._frame += 1

        # Path from the front axle, ignored while zooming in (pixels too coarse)
        path = None
        if len(trajectory) >= 2 and zoom >= geo.ZOOM / 2:
            path = geo.path_ahead(geo.to_vehicle_frame(trajectory, zoom), np.array([geo.FRONT_AXLE, 0.0]))

        # No path: let the steering decay
        if path is None:
            self._steering *= HOLD_DECAY
            self._heading_error = None
            return self._steering

        # Cross track error, positive = path on the right
        cross_track_error = path[0, 1]

        # Heading error over the chord, positive = path turns right
        k = int(np.clip(round(self.preview / geo.PATH_SPACING), 1, len(path) - 1))
        direction = path[k] - path[0]
        heading_error = np.arctan2(direction[1], direction[0])

        # Stanley law
        steering = heading_error + np.arctan2(self.gain * cross_track_error, self.softening + speed)

        # D-term on the heading error (yaw damping)
        if self._heading_error is not None:
            change = (heading_error - self._heading_error + np.pi) % (2 * np.pi) - np.pi
            steering += self.damping * change / geo.DT
        self._heading_error = heading_error

        self._steering = float(np.clip(steering, -1.0, 1.0))
        return self._steering
