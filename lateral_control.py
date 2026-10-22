import numpy as np

import geometry as geo

# Ohne Pfad klingt der letzte Lenkwinkel pro Frame mit diesem Faktor ab
HOLD_DECAY = 0.9


class LateralControl:
    """
    Querregelung mit dem Stanley-Regler.

    Gerechnet wird an der Vorderachse in metrischen Fahrzeugkoordinaten:

        delta = psi + arctan(k * e / (k_soft + v)) + k_d * dpsi/dt

    psi ist der Kurswinkelfehler, e die Querabweichung und v die Geschwindigkeit.
    Der letzte Term dämpft wie ein D-Anteil das Aufschwingen bei hohem Tempo.
    """

    def __init__(self, gain: float = 1.0, softening: float = 1.0, damping: float = 0.02, preview: float = 12.0):
        self.gain = gain  # k in 1/s, Gewicht der Querabweichung
        self.softening = softening  # k_soft in Einheiten/s, begrenzt die Verstärkung beim Anfahren
        self.damping = damping  # k_d in s, D-Anteil auf den Kurswinkelfehler
        # Länge der Sehne in Einheiten, über die die Pfadrichtung gemessen wird. Der klassische
        # Stanley nimmt die Tangente (ca. 1). Länger glättet und schneidet Kurven leicht an.
        self.preview = preview
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

        # Phase 1: Pfad ab dem Lotfußpunkt der Vorderachse.
        # Beim Einzoomen am Start ist ein Pixel noch zu grob, solange wird der Pfad ignoriert.
        path = None
        if len(trajectory) >= 2 and zoom >= geo.ZOOM / 2:
            path = geo.path_ahead(geo.to_vehicle_frame(trajectory, zoom), np.array([geo.FRONT_AXLE, 0.0]))

        # Ohne Pfad den Lenkwinkel abklingen lassen. Kurze Aussetzer in der Kurve werden so
        # überbrückt, bei längeren fährt das Auto geradeaus statt im Kreis.
        if path is None:
            self._steering *= HOLD_DECAY
            self._heading_error = None
            return self._steering

        # Phase 2: Fehler bestimmen
        # Querabweichung: seitlicher Abstand des Lotfußpunkts, positiv = Pfad liegt rechts
        cross_track_error = path[0, 1]

        # Kurswinkelfehler: Pfadrichtung relativ zur Fahrzeuglängsachse, positiv = Pfad dreht nach rechts
        k = int(np.clip(round(self.preview / geo.PATH_SPACING), 1, len(path) - 1))
        direction = path[k] - path[0]
        heading_error = np.arctan2(direction[1], direction[0])

        # Phase 3: Stanley-Regelgesetz
        steering = heading_error + np.arctan2(self.gain * cross_track_error, self.softening + speed)

        # Phase 4: D-Anteil auf den Kurswinkelfehler (wirkt wie eine Gierdämpfung)
        if self._heading_error is not None:
            change = (heading_error - self._heading_error + np.pi) % (2 * np.pi) - np.pi
            steering += self.damping * change / geo.DT
        self._heading_error = heading_error

        self._steering = float(np.clip(steering, -1.0, 1.0))
        return self._steering
