"""
Gemeinsame Geometrie für Quer- und Längsregelung.

Rechnet Bildpunkte in ein metrisches Fahrzeugkoordinatensystem um und wertet
Pfade aus. Die Konstanten stammen aus gymnasium/envs/box2d (car_racing.py und
car_dynamics.py).
"""

import numpy as np

# Die Physik läuft mit 50 FPS
DT = 1.0 / 50.0

# Kamera: Das 1000x800 Fenster wird auf 96x96 Pixel skaliert, quer und längs
# also unterschiedlich stark. Pixel pro Einheit = Zoom * Skalierung.
SCALE_X = 96 / 1000
SCALE_Y = 96 / 800
ZOOM = 2.7 * 6.0  # Zoom nach dem Einzoomen, dann 1.56 px pro Einheit quer und 1.94 längs

# Fahrzeugursprung im Bild: Fenstermitte, 3/4 der Fensterhöhe
CAR_X_PX = 48.0
CAR_Y_PX = 72.0

# Fahrzeugmaße in Einheiten (Radpositionen * SIZE aus car_dynamics.py)
FRONT_AXLE = 80 * 0.02  # Vorderachse vor dem Ursprung
REAR_AXLE = -82 * 0.02  # Hinterachse hinter dem Ursprung
WHEELBASE = FRONT_AXLE - REAR_AXLE
MAX_STEER = 0.4  # Lenkanschlag der Vorderräder in rad

# Punktabstand nach dem Neuabtasten eines Pfads
PATH_SPACING = 1.0


def camera_zoom(frame: int) -> float:
    """
    Zoom der Kamera im Frame `frame` (0 = Bild nach reset).

    Die Kamera zoomt in der ersten Sekunde von 0.6 auf ZOOM ein (car_racing.py). Das Bild
    nach reset entsteht schon bei t = DT.
    """
    t = (frame + 1) * DT
    return 0.6 * max(1.0 - t, 0.0) + ZOOM * min(t, 1.0)


def to_vehicle_frame(points, zoom: float = ZOOM) -> np.ndarray:
    """
    Rechnet Bildpunkte (x, y) in Fahrzeugkoordinaten um.

    returns:
        points (np.ndarray): (N, 2) mit Spalte 0 = nach vorne und Spalte 1 = nach rechts,
            in Einheiten der Simulation, Ursprung im Fahrzeugursprung.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    forward = (CAR_Y_PX - points[:, 1]) / (zoom * SCALE_Y)
    right = (points[:, 0] - CAR_X_PX) / (zoom * SCALE_X)
    return np.column_stack((forward, right))


def path_ahead(path: np.ndarray, origin: np.ndarray, max_gap: float = 10.0) -> np.ndarray | None:
    """
    Schneidet den Pfad am nächstgelegenen Punkt zu `origin` ab und tastet den Rest
    gleichmäßig mit PATH_SPACING neu ab.

    Der nächste Punkt ist der Lotfußpunkt auf dem nächsten Segment. Er wandert dadurch
    stetig mit und springt nicht von Stützpunkt zu Stützpunkt. Segmente länger als
    `max_gap` gelten als Lücke und beenden den Pfad (z. B. Start/Ziel in der Ground Truth
    oder ein anderer Streckenabschnitt im Bild).

    returns:
        path (np.ndarray | None): (M, 2) gleichmäßig abgetastet, der erste Punkt ist der
            Lotfußpunkt. None, wenn weniger als zwei Punkte übrig bleiben.
    """
    start, end = path[:-1], path[1:]
    segment = end - start
    length = np.hypot(segment[:, 0], segment[:, 1])
    gap = length > max_gap
    usable = ~gap & (length > 1e-9)
    if not usable.any():
        return None

    # Lotfußpunkt auf jedem Segment, davon den nächsten zum Ursprung
    t = np.einsum("ij,ij->i", origin - start, segment) / np.maximum(length, 1e-9) ** 2
    foot = start + np.clip(t, 0.0, 1.0)[:, None] * segment
    distance = np.hypot(foot[:, 0] - origin[0], foot[:, 1] - origin[1])
    distance[~usable] = np.inf
    i = int(np.argmin(distance))

    # Vom Lotfußpunkt bis zur ersten Lücke
    next_gap = np.flatnonzero(gap[i + 1:])
    last = i + 1 + (next_gap[0] if next_gap.size else len(segment) - i - 1)
    points = np.vstack((foot[i], path[i + 1:last + 1]))

    # Neu abtasten über die Bogenlänge
    step = np.hypot(*np.diff(points, axis=0).T)
    arc = np.concatenate(([0.0], np.cumsum(step)))
    samples = np.arange(0.0, arc[-1], PATH_SPACING)
    if samples.size < 2:
        return None
    return np.column_stack((np.interp(samples, arc, points[:, 0]), np.interp(samples, arc, points[:, 1])))


def curvature(path: np.ndarray, window: int) -> np.ndarray:
    """
    Krümmung eines gleichmäßig abgetasteten Pfads in 1/Einheit, positiv = Rechtskurve.

    Kreis durch je drei Punkte im Abstand `window` (Menger-Krümmung). Ein größeres
    Fenster glättet das Rauschen der Pixelpunkte.

    returns:
        curvature (np.ndarray): Länge M - 2 * window, Wert j gehört zum Pfadpunkt j + window.
    """
    a, b, c = path[:-2 * window], path[window:-window], path[2 * window:]
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    norm = np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ac.T)
    return 2.0 * cross / np.maximum(norm, 1e-9)
