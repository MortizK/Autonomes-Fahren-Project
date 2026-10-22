import numpy as np

import geometry as geo

# Ohne Pfad klingt der letzte Lenkwinkel pro Frame mit diesem Faktor ab
HOLD_DECAY = 0.9


class LateralControl:
    """
    Querregelung, standardmäßig mit dem Stanley-Regler.

    Stanley rechnet an der Vorderachse in metrischen Fahrzeugkoordinaten:

        delta = psi + arctan(k * e / (k_soft + v)) + k_d * dpsi/dt

    psi ist der Kurswinkelfehler, e die Querabweichung und v die Geschwindigkeit.
    Der letzte Term dämpft wie ein D-Anteil das Aufschwingen bei hohem Tempo.

    Zum Vergleich gibt es Pure Pursuit (method="pure_pursuit"): Von der Hinterachse aus
    wird ein Kreisbogen durch einen Punkt im Abstand l_d auf dem Pfad gelegt,

        delta = arctan(2 * L * sin(alpha) / l_d),  l_d = max(k_v * v, l_min)
    """

    def __init__(
        self,
        method: str = "stanley",
        gain: float = 1.0,
        softening: float = 1.0,
        damping: float = 0.02,
        preview: float = 12.0,
        lookahead_gain: float = 0.25,
        lookahead_min: float = 6.0,
    ):
        self.method = method  # "stanley" oder "pure_pursuit"

        # Stanley
        self.gain = gain  # k in 1/s, Gewicht der Querabweichung
        self.softening = softening  # k_soft in Einheiten/s, begrenzt die Verstärkung beim Anfahren
        self.damping = damping  # k_d in s, D-Anteil auf den Kurswinkelfehler
        # Länge der Sehne in Einheiten, über die die Pfadrichtung gemessen wird. Der klassische
        # Stanley nimmt die Tangente (ca. 1). Länger glättet und schneidet Kurven leicht an.
        self.preview = preview

        # Pure Pursuit
        self.lookahead_gain = lookahead_gain  # k_v in s, Vorausschau wächst mit dem Tempo
        self.lookahead_min = lookahead_min  # l_min in Einheiten

        self.reset()

    def reset(self) -> None:
        """Setzt den Reglerzustand zurück, z. B. bei einer neuen Episode."""
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
            steering_angle (float): Soll-Radwinkel in rad, positiv = rechts. Die Vorderräder
                stehen ab +/-0.4 rad am Anschlag, größere Werte lenken nur schneller ein.
        """
        zoom = geo.camera_zoom(self._frame)
        self._frame += 1

        # Beim Einzoomen am Start ist ein Pixel noch zu grob, solange wird der Pfad ignoriert
        steering = None
        if len(trajectory) >= 2 and zoom >= geo.ZOOM / 2:
            points = geo.to_vehicle_frame(trajectory, zoom)
            if self.method == "pure_pursuit":
                steering = self._pure_pursuit(points, speed)
            else:
                steering = self._stanley(points, speed)

        # Ohne Pfad den Lenkwinkel abklingen lassen. Kurze Aussetzer in der Kurve werden so
        # überbrückt, bei längeren fährt das Auto geradeaus statt im Kreis.
        if steering is None:
            self._steering *= HOLD_DECAY
            self._heading_error = None
            return self._steering

        self._steering = float(np.clip(steering, -1.0, 1.0))
        return self._steering

    def _stanley(self, points: np.ndarray, speed: float) -> float | None:
        """Stanley-Regler auf Pfadpunkten in Fahrzeugkoordinaten, None ohne gültigen Pfad."""
        # Pfad ab dem Lotfußpunkt der Vorderachse
        path = geo.path_ahead(points, np.array([geo.FRONT_AXLE, 0.0]))
        if path is None:
            return None

        # Querabweichung: seitlicher Abstand des Lotfußpunkts, positiv = Pfad liegt rechts
        cross_track_error = path[0, 1]

        # Kurswinkelfehler: Pfadrichtung relativ zur Fahrzeuglängsachse, positiv = Pfad dreht nach rechts
        k = int(np.clip(round(self.preview / geo.PATH_SPACING), 1, len(path) - 1))
        direction = path[k] - path[0]
        heading_error = np.arctan2(direction[1], direction[0])

        steering = heading_error + np.arctan2(self.gain * cross_track_error, self.softening + speed)

        # D-Anteil auf den Kurswinkelfehler (wirkt wie eine Gierdämpfung)
        if self._heading_error is not None:
            change = (heading_error - self._heading_error + np.pi) % (2 * np.pi) - np.pi
            steering += self.damping * change / geo.DT
        self._heading_error = heading_error

        return steering

    def _pure_pursuit(self, points: np.ndarray, speed: float) -> float | None:
        """Pure Pursuit auf Pfadpunkten in Fahrzeugkoordinaten, None ohne gültigen Pfad."""
        rear_axle = np.array([geo.REAR_AXLE, 0.0])
        path = geo.path_ahead(points, rear_axle)
        if path is None:
            return None

        # Erster Pfadpunkt außerhalb der Vorausschau, sonst das Pfadende
        offset = path - rear_axle
        distance = np.hypot(offset[:, 0], offset[:, 1])
        beyond = distance >= max(self.lookahead_gain * speed, self.lookahead_min)
        i = int(np.argmax(beyond)) if beyond.any() else len(path) - 1

        # alpha: Winkel zum Zielpunkt relativ zur Fahrzeuglängsachse
        alpha = np.arctan2(offset[i, 1], offset[i, 0])
        return np.arctan2(2.0 * geo.WHEELBASE * np.sin(alpha), distance[i])
