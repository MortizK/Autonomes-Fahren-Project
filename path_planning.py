"""Local, vehicle-aware geometric racing-line planning.

The public :meth:`PathPlanning.plan` adapter intentionally remains compatible
with the original pipeline.  All geometry stays in image coordinates; the
controllers perform the image-to-vehicle conversion.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanningResult:
    trajectory: np.ndarray
    offsets: np.ndarray
    curvature: np.ndarray
    valid: bool
    status: str
    optimizer_success: bool = False
    optimizer_message: str = ""


class PathPlanning:
    """Plan a smooth local racing line inside two ordered image polylines."""

    def __init__(
        self,
        vehicle_width: float = 4.0,
        vehicle_length: float = 8.0,
        safety_margin: float = 1.0,
        sample_spacing: float = 3.0,
        curvature_weight: float = 1.0,
        curvature_smoothness_weight: float = 0.08,
        continuity_weight: float = 0.05,
        control_points: int = 16,
        smoothing_iterations: int = 4,
        racing_line_gain: float = 0.35,
    ):
        if not all(
            np.isfinite(value) and value >= 0.0
            for value in (
                vehicle_width,
                vehicle_length,
                safety_margin,
                sample_spacing,
                curvature_weight,
                curvature_smoothness_weight,
                continuity_weight,
                racing_line_gain,
            )
        ) or sample_spacing <= 0.0 or control_points < 4 or smoothing_iterations < 1:
            raise ValueError("planner dimensions and weights must be finite and non-negative")
        self.vehicle_width = float(vehicle_width)
        self.vehicle_length = float(vehicle_length)
        self.safety_margin = float(safety_margin)
        self.sample_spacing = float(sample_spacing)
        self.curvature_weight = float(curvature_weight)
        self.curvature_smoothness_weight = float(curvature_smoothness_weight)
        self.continuity_weight = float(continuity_weight)
        self.control_points = int(control_points)
        self.smoothing_iterations = int(smoothing_iterations)
        self.racing_line_gain = float(racing_line_gain)
        self._previous_offsets: np.ndarray | None = None

    @staticmethod
    def _polyline(points) -> np.ndarray | None:
        try:
            array = np.asarray(points, dtype=float)
        except (TypeError, ValueError):
            return None
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
            return None
        if not np.isfinite(array).all():
            return None
        keep = np.r_[True, np.linalg.norm(np.diff(array, axis=0), axis=1) > 1e-9]
        array = array[keep]
        return array if len(array) >= 2 else None

    @staticmethod
    def _arc(points: np.ndarray) -> tuple[np.ndarray, float] | None:
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if not np.isfinite(lengths).all() or lengths.sum() <= 1e-9:
            return None
        return np.r_[0.0, np.cumsum(lengths)], float(lengths.sum())

    @staticmethod
    def _interpolate(points: np.ndarray, arc: np.ndarray, values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.interp(values, arc, points[:, axis]) for axis in range(2)]
        )

    @staticmethod
    def _curvature(path: np.ndarray) -> np.ndarray:
        result = np.zeros(len(path), dtype=float)
        if len(path) < 3:
            return result
        first = path[:-2]
        middle = path[1:-1]
        last = path[2:]
        a = middle - first
        b = last - middle
        c = last - first
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        denominator = (
            np.linalg.norm(a, axis=1)
            * np.linalg.norm(b, axis=1)
            * np.linalg.norm(c, axis=1)
        )
        result[1:-1] = np.divide(
            2.0 * cross,
            denominator,
            out=np.zeros_like(cross),
            where=denominator > 1e-9,
        )
        if len(result) > 2:
            result[0] = result[1]
            result[-1] = result[-2]
        return result

    @staticmethod
    def _segments_intersect(a, b, c, d) -> bool:
        def orient(p, q, r):
            value = np.cross(q - p, r - p)
            return float(value)

        eps = 1e-8
        ab_c = orient(a, b, c)
        ab_d = orient(a, b, d)
        cd_a = orient(c, d, a)
        cd_b = orient(c, d, b)
        if (
            max(min(a[0], b[0]), min(c[0], d[0]))
            > min(max(a[0], b[0]), max(c[0], d[0])) + eps
            or max(min(a[1], b[1]), min(c[1], d[1]))
            > min(max(a[1], b[1]), max(c[1], d[1])) + eps
        ):
            return False
        return (ab_c * ab_d <= eps) and (cd_a * cd_b <= eps)

    @classmethod
    def _simple_polygon(cls, polygon: np.ndarray) -> bool:
        if len(polygon) < 4:
            return False
        segments = list(zip(polygon, np.roll(polygon, -1, axis=0)))
        for i, (a, b) in enumerate(segments):
            for j in range(i + 1, len(segments)):
                if j in (i - 1, i + 1) or (i == 0 and j == len(segments) - 1):
                    continue
                if cls._segments_intersect(a, b, *segments[j]):
                    return False
        return True

    @staticmethod
    def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        x = points[:, 0]
        y = points[:, 1]
        inside = np.zeros(len(points), dtype=bool)
        on_boundary = np.zeros(len(points), dtype=bool)
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            crosses = (start[1] > y) != (end[1] > y)
            denominator = end[1] - start[1]
            safe_denominator = denominator if abs(denominator) > 1e-12 else 1e-12
            crossing_x = (end[0] - start[0]) * (y - start[1]) / safe_denominator + start[0]
            inside ^= crosses & (x < crossing_x)
            segment = end - start
            length_squared = float(np.dot(segment, segment))
            if length_squared > 1e-12:
                projection = np.clip(
                    ((points - start) @ segment) / length_squared, 0.0, 1.0
                )
                distance = np.linalg.norm(points - (start + projection[:, None] * segment), axis=1)
                on_boundary |= distance <= 1e-7
        return inside | on_boundary

    def _vehicle_inside(
        self, path: np.ndarray, tangents: np.ndarray, polygon: np.ndarray
    ) -> tuple[bool, int]:
        half_width = self.vehicle_width / 2.0 + self.safety_margin
        half_length = self.vehicle_length / 2.0
        path_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        total_length = path_arc[-1]
        spacing = max(float(np.median(np.diff(path_arc))), 1e-6)
        endpoint_samples = int(np.ceil(half_length / spacing))
        for index, (point, tangent) in enumerate(zip(path[1:], tangents[1:]), start=1):
            if index <= endpoint_samples or index >= len(path) - endpoint_samples - 1:
                continue
            lateral = np.array([-tangent[1], tangent[0]])
            backward = min(half_length, path_arc[index])
            forward = min(half_length, total_length - path_arc[index])
            if backward + forward <= 1e-9:
                continue
            longitudinal = (-backward, forward)
            corners = np.array(
                [
                    point + longitudinal[1] * tangent + half_width * lateral,
                    point + longitudinal[1] * tangent - half_width * lateral,
                    point + longitudinal[0] * tangent - half_width * lateral,
                    point + longitudinal[0] * tangent + half_width * lateral,
                ]
            )
            samples = np.vstack(
                [
                    corners,
                    (corners + np.roll(corners, -1, axis=0)) / 2.0,
                ]
            )
            if not self._inside(samples, polygon).all():
                return False, index
            if index:
                previous = path[index - 1]
                for fraction in (0.25, 0.5, 0.75):
                    intermediate = previous * (1.0 - fraction) + point * fraction
                    if not self._inside(intermediate[None], polygon)[0]:
                        return False, index
        return True, -1

    def _prepare(self, left, right):
        left_points = self._polyline(left)
        right_points = self._polyline(right)
        if left_points is None or right_points is None:
            return None
        left_arc = self._arc(left_points)
        right_arc = self._arc(right_points)
        if left_arc is None or right_arc is None:
            return None
        left_lengths, left_total = left_arc
        right_lengths, right_total = right_arc
        count = max(12, int(np.ceil(max(left_total, right_total) / self.sample_spacing)) + 1)
        fractions = np.linspace(0.0, 1.0, count)
        left_samples = self._interpolate(left_points, left_lengths, fractions * left_total)
        right_samples = self._interpolate(right_points, right_lengths, fractions * right_total)
        center = (left_samples + right_samples) / 2.0
        center_arc = self._arc(center)
        if center_arc is None:
            return None
        center_lengths, center_total = center_arc
        if center_total < self.sample_spacing * 2.0:
            return None
        count = max(12, int(np.floor(center_total / self.sample_spacing)) + 1)
        values = np.linspace(0.0, center_total, count)
        center = self._interpolate(center, center_lengths, values)
        left_samples = self._interpolate(left_samples, center_lengths, values)
        right_samples = self._interpolate(right_samples, center_lengths, values)
        delta = np.gradient(center, axis=0)
        norms = np.linalg.norm(delta, axis=1)
        if (norms <= 1e-8).any():
            return None
        tangents = delta / norms[:, None]
        normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
        left_offsets = np.einsum("ij,ij->i", left_samples - center, normals)
        right_offsets = np.einsum("ij,ij->i", right_samples - center, normals)
        if not (np.all(left_offsets > 0.0) and np.all(right_offsets < 0.0)):
            return None
        if np.any(left_offsets - right_offsets <= self.vehicle_width + 2.0 * self.safety_margin):
            return None
        polygon = np.vstack((left_points, right_points[::-1]))
        return center, tangents, normals, left_samples, right_samples, left_offsets, right_offsets, polygon

    def _plan(self, left_boundary, right_boundary, validate_footprint: bool) -> PlanningResult:
        prepared = self._prepare(left_boundary, right_boundary)
        if prepared is None:
            self._previous_offsets = None
            return PlanningResult(
                np.empty((0, 2)), np.empty(0), np.empty(0), False, "invalid geometry"
            )
        (
            center,
            tangents,
            normals,
            left_samples,
            right_samples,
            left_offsets,
            right_offsets,
            polygon,
        ) = prepared
        center_curvature = self._curvature(center)
        half_width = self.vehicle_width / 2.0 + self.safety_margin
        length_clearance = 0.25 * self.vehicle_length * np.abs(center_curvature)
        lower = right_offsets + half_width + length_clearance
        upper = left_offsets - half_width - length_clearance
        if np.any(lower >= upper):
            self._previous_offsets = None
            return PlanningResult(
                np.empty((0, 2)), np.empty(0), np.empty(0), False, "corridor too narrow"
            )
        initial = np.clip(np.zeros(len(center)), lower, upper)
        previous = self._previous_offsets
        if previous is not None and len(previous) == len(initial) and np.isfinite(previous).all():
            initial = np.clip(previous, lower, upper)
        
        def path_for(offset_profile):
            return center + offset_profile[:, None] * normals

        controls = min(self.control_points, len(center))
        control_indices = np.linspace(0, len(center) - 1, controls).round().astype(int)
        control_lower = lower[control_indices]
        control_upper = upper[control_indices]
        control_initial = initial[control_indices]
        control_curvature = center_curvature[control_indices]
        available_width = np.minimum(
            left_offsets[control_indices], -right_offsets[control_indices]
        )
        desired = -np.tanh(30.0 * control_curvature)
        desired *= self.racing_line_gain * np.maximum(available_width - half_width, 0.0)
        offsets = np.clip(control_initial, control_lower, control_upper)
        for _ in range(self.smoothing_iterations):
            smoothed = offsets.copy()
            if controls > 2:
                smoothed[1:-1] = (
                    0.5 * offsets[1:-1]
                    + 0.25 * (offsets[:-2] + offsets[2:])
                    + 0.25 * desired[1:-1]
                )
            offsets = np.clip(smoothed, control_lower, control_upper)
        sample_positions = np.linspace(0.0, controls - 1, len(center))
        offsets = np.interp(sample_positions, np.arange(controls), offsets)
        offsets = np.clip(offsets, lower, upper)
        path = path_for(offsets)
        curvature = self._curvature(path)
        finite = np.isfinite(path).all() and np.isfinite(curvature).all()
        if validate_footprint and not self._simple_polygon(polygon):
            finite = False
            status = "invalid corridor polygon"
            failing_index = -1
        elif validate_footprint:
            valid_footprint, failing_index = self._vehicle_inside(path, tangents, polygon)
            finite = finite and valid_footprint
            status = "vehicle footprint collision"
        else:
            failing_index = -1
            status = "invalid trajectory"
        if not finite:
            self._previous_offsets = None
            if failing_index >= 0:
                status += f" at sample {failing_index}"
            return PlanningResult(
                np.empty((0, 2)),
                np.empty(0),
                np.empty(0),
                False,
                status,
            )
        self._previous_offsets = offsets.copy()
        return PlanningResult(
            path,
            offsets,
            curvature,
            True,
            "ok",
        )

    def plan_detailed(self, left_boundary, right_boundary) -> PlanningResult:
        """Return a fully validated local plan and diagnostics without a GUI."""
        return self._plan(left_boundary, right_boundary, validate_footprint=True)

    def plan(self, left_lane_boundaries: list, right_lane_boundaries: list) -> tuple[list, list]:
        """Return image-space trajectory and finite diagnostic curvature lists."""
        result = self._plan(left_lane_boundaries, right_lane_boundaries, validate_footprint=False)
        if not result.valid:
            return [], []
        return result.trajectory.tolist(), result.curvature.tolist()
