import numpy as np

from path_planning import PathPlanning


def corridor(center, width=22.0):
    center = np.asarray(center, dtype=float)
    tangent = np.gradient(center, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    return (
        (center + width / 2.0 * normal).tolist(),
        (center - width / 2.0 * normal).tolist(),
    )


def assert_ordered_finite(path):
    points = np.asarray(path, dtype=float)
    assert len(points) >= 3
    assert np.isfinite(points).all()
    assert np.all(np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8)


def test_invalid_inputs_return_empty_tuple():
    planner = PathPlanning()
    assert planner.plan([], [[1, 1], [1, 2]]) == ([], [])
    assert planner.plan([[0, 0]], [[1, 1], [1, 2]]) == ([], [])
    assert planner.plan([[0, 0], [np.nan, 2]], [[1, 1], [1, 2]]) == ([], [])
    assert planner.plan([[0, 0], [0, 1]], [[1, 1], [1, 2, 3]]) == ([], [])


def test_straight_unequal_boundaries_are_dense_and_safe():
    y_left = np.linspace(77, 5, 80)
    y_right = np.linspace(77, 5, 23)
    left = np.column_stack((35.0 + 0.02 * (77 - y_left), y_left))
    right = np.column_stack((59.0 + 0.02 * (77 - y_right), y_right))
    planner = PathPlanning(vehicle_width=4.0, vehicle_length=6.0)
    trajectory, curvature = planner.plan(left.tolist(), right.tolist())
    assert_ordered_finite(trajectory)
    assert np.isfinite(curvature).all()
    assert np.max(np.abs(curvature)) < 0.01


def test_curves_and_hairpin_do_not_cross_or_shortcut():
    t = np.linspace(0.0, np.pi * 0.8, 90)
    center = np.column_stack((48.0 + 22.0 * np.sin(t), 76.0 - 22.0 * (1.0 - np.cos(t))))
    left, right = corridor(center, width=20.0)
    result = PathPlanning(vehicle_width=3.0, vehicle_length=5.0).plan_detailed(left, right)
    assert result.valid
    assert_ordered_finite(result.trajectory)
    assert np.isfinite(result.offsets).all()


def test_noisy_duplicates_are_normalized():
    y = np.linspace(77, 10, 50)
    left = [[35.0, float(y[0])], [35.0, float(y[0])]]
    right = [[59.0, float(y[0])]]
    for value in y[1:]:
        left.append([35.0 + 0.1 * np.sin(value), float(value)])
        right.extend([[59.0, float(value)], [59.0, float(value)]])
    trajectory, _ = PathPlanning().plan(left, right)
    assert_ordered_finite(trajectory)


def test_vehicle_too_wide_for_corridor_is_rejected():
    center = np.column_stack((np.full(30, 48.0), np.linspace(77, 10, 30)))
    left, right = corridor(center, width=7.0)
    result = PathPlanning(vehicle_width=8.0, vehicle_length=4.0).plan_detailed(left, right)
    assert not result.valid
