"""Step-by-step telemetry recording for controller debugging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import geometry as geo


def _json_value(value: Any) -> Any:
    """Convert NumPy and Box2D values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "x") and hasattr(value, "y"):
        return [float(value.x), float(value.y)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _wheel_record(wheel: Any) -> dict[str, Any]:
    forward = wheel.GetWorldVector((0, 1))
    side = wheel.GetWorldVector((1, 0))
    velocity = wheel.linearVelocity
    forward_speed = forward[0] * velocity[0] + forward[1] * velocity[1]
    side_speed = side[0] * velocity[0] + side[1] * velocity[1]
    road_friction = max(
        (getattr(tile, "road_friction", 0.0) for tile in wheel.tiles),
        default=0.0,
    )

    return {
        "omega": float(wheel.omega),
        "tire_speed": float(wheel.omega * wheel.wheel_rad),
        "forward_speed": float(forward_speed),
        "side_speed": float(side_speed),
        "speed": float(np.linalg.norm(velocity)),
        "gas": float(wheel.gas),
        "brake": float(wheel.brake),
        "steer": float(wheel.steer),
        "grip": float(road_friction if road_friction else 0.6),
        "on_road": bool(wheel.tiles),
    }


class TelemetryLogger:
    """Writes one self-contained JSON record for every simulation step."""

    def __init__(self, directory: str | Path = "telemetry"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._directory = directory

    def start_episode(self, episode: int, seed: int) -> None:
        self.close()
        path = self._directory / f"episode_{episode:03d}_seed_{seed:06d}.jsonl"
        self._file = path.open("w", encoding="utf-8")

    def record(
        self,
        *,
        step: int,
        episode: int,
        seed: int,
        info: dict[str, Any],
        action: list[float],
        reward: float,
        terminated: bool,
        truncated: bool,
        env: Any,
        car: Any = None,
    ) -> None:
        if self._file is None:
            raise RuntimeError("start_episode() must be called before record()")

        vehicle = env.unwrapped.car
        hull = vehicle.hull
        record = {
            "dt": geo.DT,
            "episode": episode,
            "seed": seed,
            "step": step,
            "time": step * geo.DT,
            "action": {
                "steering": float(action[0]),
                "throttle": float(action[1]),
                "brake": float(action[2]),
            },
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "vehicle": {
                "speed": float(info.get("speed", 0.0)),
                "position": _json_value(hull.position),
                "velocity": _json_value(hull.linearVelocity),
                "angle": float(hull.angle),
                "angular_velocity": float(hull.angularVelocity),
                "fuel_spent": float(vehicle.fuel_spent),
                "wheels": [_wheel_record(wheel) for wheel in vehicle.wheels],
            },
        }
        if car is not None:
            longitudinal = car._longitudinal_control
            lateral = car._lateral_control
            record["controller"] = {
                "target_speed": float(longitudinal._target_speed),
                "speed_integral": float(longitudinal._integral),
                "last_speed": _json_value(longitudinal._last_speed),
                "steering": float(lateral._steering),
                "heading_error": _json_value(lateral._heading_error),
            }

        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> TelemetryLogger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
