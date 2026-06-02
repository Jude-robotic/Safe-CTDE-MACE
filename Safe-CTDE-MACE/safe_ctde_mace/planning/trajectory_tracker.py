from __future__ import annotations

from typing import Iterable

import numpy as np

from safe_ctde_mace.planning.ego_planner import ContinuousTrajectory


class TrajectoryTracker:
    """Move one discrete step along a planned voxel path."""

    def step(
        self,
        current_position: Iterable[int],
        path: list[tuple[int, int, int]] | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        current = np.asarray(tuple(current_position), dtype=float)
        if not path or len(path) == 1:
            return current, np.zeros(3, dtype=float), 0.0
        next_position = np.asarray(path[1], dtype=float)
        velocity = next_position - current
        return next_position, velocity, float(np.linalg.norm(velocity))

    def step_continuous(
        self,
        current_position: Iterable[float],
        trajectory: ContinuousTrajectory | None,
        step_duration: float,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        current = np.asarray(tuple(current_position), dtype=float)
        if trajectory is None:
            return current, np.zeros(3, dtype=float), 0.0, np.stack([current, current]), np.zeros(3)
        end_time = min(float(step_duration), trajectory.duration)
        next_position, velocity, acceleration = trajectory.sample(end_time)
        segment = trajectory.sample_segment(end_time)
        step_distance = float(np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)))
        return next_position, velocity, step_distance, segment, acceleration
