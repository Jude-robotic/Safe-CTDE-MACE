from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EpisodeMetrics:
    coverage_ratio: float = 0.0
    repeated_coverage_ratio: float = 0.0
    collision_count: int = 0
    obstacle_collision_count: int = 0
    inter_uav_collision_count: int = 0
    average_path_length: float = 0.0
    episode_length: int = 0
    communication_links: int = 0
    success: bool = False
    mean_acceleration: float = 0.0
    max_acceleration: float = 0.0
    smoothness_cost: float = 0.0


def repeated_coverage_ratio(repeated_coverage: int, total_observations: int) -> float:
    if total_observations <= 0:
        return 0.0
    return repeated_coverage / total_observations


def trajectory_metrics_from_points(points: np.ndarray, dt: float = 1.0) -> dict[str, float]:
    """Compute simple kinematic metrics from an ordered point sequence."""
    positions = np.asarray(points, dtype=float)
    if len(positions) < 2:
        return {
            "path_length": 0.0,
            "mean_acceleration": 0.0,
            "max_acceleration": 0.0,
            "smoothness_cost": 0.0,
        }
    velocities = np.diff(positions, axis=0) / max(float(dt), 1e-6)
    accelerations = np.diff(velocities, axis=0) / max(float(dt), 1e-6)
    jerks = np.diff(accelerations, axis=0) / max(float(dt), 1e-6)
    acceleration_norms = np.linalg.norm(accelerations, axis=1) if len(accelerations) else np.zeros(1)
    jerk_norms = np.linalg.norm(jerks, axis=1) if len(jerks) else np.zeros(1)
    return {
        "path_length": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        "mean_acceleration": float(np.mean(acceleration_norms)),
        "max_acceleration": float(np.max(acceleration_norms)),
        "smoothness_cost": float(np.mean(acceleration_norms**2) + np.mean(jerk_norms**2)),
    }
