"""Geometry shared by the controllers. Constants from gymnasium/envs/box2d."""

import numpy as np

DT = 1.0 / 50.0  # physics step

# 1000x800 window scaled to 96x96, so x and y are scaled differently
SCALE_X = 96 / 1000
SCALE_Y = 96 / 800
ZOOM = 2.7 * 6.0  # camera zoom after the first second

# Car origin in the image
CAR_X_PX = 48.0
CAR_Y_PX = 72.0

# Car dimensions in units
FRONT_AXLE = 80 * 0.02
REAR_AXLE = -82 * 0.02
WHEELBASE = FRONT_AXLE - REAR_AXLE
MAX_STEER = 0.4  # steering lock in rad

PATH_SPACING = 1.0  # point spacing after resampling


def camera_zoom(frame: int) -> float:
    """Camera zoom in a frame (0 = observation after reset)."""
    t = (frame + 1) * DT
    return 0.6 * max(1.0 - t, 0.0) + ZOOM * min(t, 1.0)


def to_vehicle_frame(points, zoom: float = ZOOM) -> np.ndarray:
    """Image points (x, y) to vehicle frame (forward, right) in units."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    forward = (CAR_Y_PX - points[:, 1]) / (zoom * SCALE_Y)
    right = (points[:, 0] - CAR_X_PX) / (zoom * SCALE_X)
    return np.column_stack((forward, right))


def path_ahead(path: np.ndarray, origin: np.ndarray, max_gap: float = 10.0) -> np.ndarray | None:
    """
    Path from the closest point to `origin` up to the first gap, resampled.

    returns:
        path (np.ndarray | None): (M, 2), starts at the closest point. None if shorter than two points.
    """
    start, end = path[:-1], path[1:]
    segment = end - start
    length = np.hypot(segment[:, 0], segment[:, 1])
    gap = length > max_gap
    usable = ~gap & (length > 1e-9)
    if not usable.any():
        return None

    # Closest point on each segment
    t = np.einsum("ij,ij->i", origin - start, segment) / np.maximum(length, 1e-9) ** 2
    foot = start + np.clip(t, 0.0, 1.0)[:, None] * segment
    distance = np.hypot(foot[:, 0] - origin[0], foot[:, 1] - origin[1])
    distance[~usable] = np.inf
    i = int(np.argmin(distance))

    # Up to the next gap
    next_gap = np.flatnonzero(gap[i + 1:])
    last = i + 1 + (next_gap[0] if next_gap.size else len(segment) - i - 1)
    points = np.vstack((foot[i], path[i + 1:last + 1]))

    # Resample by arc length
    step = np.hypot(*np.diff(points, axis=0).T)
    arc = np.concatenate(([0.0], np.cumsum(step)))
    samples = np.arange(0.0, arc[-1], PATH_SPACING)
    if samples.size < 2:
        return None
    return np.column_stack((np.interp(samples, arc, points[:, 0]), np.interp(samples, arc, points[:, 1])))


def curvature(path: np.ndarray, window: int) -> np.ndarray:
    """
    Signed curvature in 1/unit (positive = right), circle through points `window` apart.

    returns:
        curvature (np.ndarray): value j belongs to path point j + window.
    """
    a, b, c = path[:-2 * window], path[window:-window], path[2 * window:]
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    norm = np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ac.T)
    return 2.0 * cross / np.maximum(norm, 1e-9)
