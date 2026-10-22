import numpy as np

import geometry as geo

# Gemessene Kennwerte des Autos auf Asphalt in Einheiten/s^2
GAS_ACCEL = 44.0  # Beschleunigung bei Vollgas, die Hinterräder drehen dabei schon durch
BRAKE_ACCEL = 300.0  # Verzögerung pro Bremswert, linear bis ca. 0.7
GRIP_ACCEL = 219.0  # Haftgrenze der Reifen (Verzögerung mit blockierten Rädern)


class LongitudinalControl:
    """
    Längsregelung in zwei Schritten:

    1. Zielgeschwindigkeit aus der Krümmung des Pfads. Jede Kurve begrenzt das Tempo über
       die erlaubte Querbeschleunigung, v = sqrt(a_quer / kappa). Weiter entfernte Kurven
       erlauben mehr, solange bis dorthin noch abgebremst werden kann,
       v = sqrt(v_kurve^2 + 2 * a_brems * s).
    2. PID-Regler, der aus dem Geschwindigkeitsfehler eine Soll-Beschleunigung macht. Über
       die gemessenen Kennwerte wird daraus Gas oder Bremse.
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
        # PID, Ausgang ist eine Soll-Beschleunigung
        self.kp = kp  # 1/s
        self.ki = ki  # 1/s^2
        self.kd = kd  # dimensionslos, wirkt auf die gemessene Beschleunigung

        # Zielgeschwindigkeit
        self.max_speed = max_speed
        self.min_speed = min_speed
        # Bezogen auf die Mittellinie. Liegt über der Haftgrenze, weil das Auto kurze Kurven etwas schneidet.
        self.lateral_accel = lateral_accel
        self.brake_accel = brake_accel  # eingeplante Verzögerung vor Kurven, mit Reserve zum Maximum
        # Punkte links und rechts für die Krümmung. Kurze Kurven wirken damit flacher als lange.
        self.curvature_window = curvature_window

        # Ab 0.9 blockieren die Räder, ab ca. 0.7 ist die Haftung ausgereizt
        self.max_brake = max_brake

        self.reset()

    def reset(self) -> None:
        """Setzt den Reglerzustand zurück, z. B. bei einer neuen Episode."""
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

        # Beim Einzoomen am Start ist ein Pixel noch zu grob, solange wird der Pfad ignoriert
        path = None
        if len(trajectory) >= 2 and zoom >= geo.ZOOM / 2:
            path = geo.path_ahead(geo.to_vehicle_frame(trajectory, zoom), np.zeros(2))

        # Ohne Pfad so abbremsen, wie es vor einer Kurve eingeplant ist
        if path is None or len(path) <= 2 * self.curvature_window:
            self._target_speed = max(self._target_speed - self.brake_accel * geo.DT, self.min_speed)
            return self._target_speed

        # Kurvengeschwindigkeit an jedem Pfadpunkt
        curvature = np.abs(geo.curvature(path, self.curvature_window))
        corner_speed = np.sqrt(self.lateral_accel / np.maximum(curvature, 1e-6))

        # Geschwindigkeit, von der aus bis zu diesem Punkt noch auf die Kurvengeschwindigkeit gebremst werden kann.
        # Die Krümmung direkt am Auto fehlt (Fenster), daher gilt der erste Wert schon ab dem Auto.
        distance = np.arange(curvature.size) * geo.PATH_SPACING
        reachable = np.sqrt(corner_speed**2 + 2 * self.brake_accel * distance)

        # Hinter dem Sichtbereich kann eine enge Kurve kommen, bis zum Pfadende muss man auf min_speed kommen
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

        # D-Anteil auf die gemessene Beschleunigung statt auf den Fehler,
        # damit Sprünge der Zielgeschwindigkeit keinen Stoß erzeugen
        if self._last_speed is None:
            self._last_speed = current_speed
        accel = (current_speed - self._last_speed) / geo.DT
        self._last_speed = current_speed

        demand = self.kp * error + self.ki * self._integral - self.kd * accel

        # Anti-Windup: nur integrieren, solange der Ausgang nicht in der Begrenzung steckt
        # oder der Fehler wieder aus ihr herausführt
        saturated = demand > GAS_ACCEL or demand < -self.max_brake * BRAKE_ACCEL
        if not saturated or np.sign(error) != np.sign(demand):
            self._integral += error * geo.DT

        # Kammscher Kreis: Querbeschleunigung aus Lenkwinkel und Tempo (Einspurmodell),
        # nur die restliche Haftung darf für Gas genutzt werden. Sonst brechen die Hinterräder aus.
        lateral = current_speed**2 * np.tan(min(abs(steering_angle), geo.MAX_STEER)) / geo.WHEELBASE
        grip = np.sqrt(max(1.0 - (lateral / GRIP_ACCEL) ** 2, 0.0))

        # Bremsen bleibt auch in der Kurve erlaubt, zu schnell in der Kurve ist schlimmer
        if demand >= 0.0:
            throttle = min(demand / GAS_ACCEL, grip)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-demand / BRAKE_ACCEL, self.max_brake)

        return float(throttle), float(brake)
