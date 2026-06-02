from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt

from safe_ctde_mace.mapping.voxel_map import TRAVERSABLE_STATES, VoxelState
from safe_ctde_mace.planning.astar_3d import AStar3D


@dataclass(slots=True)
class ContinuousTrajectory:
    """Sampleable continuous local trajectory."""

    times: np.ndarray
    waypoints: np.ndarray
    sample_dt: float
    spline: CubicSpline | None = None

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @classmethod
    def from_waypoints(
        cls,
        waypoints: np.ndarray,
        max_velocity: float,
        sample_dt: float,
        smooth: bool = True,
    ) -> "ContinuousTrajectory":
        points = np.asarray(waypoints, dtype=float)
        if len(points) < 2:
            points = np.vstack([points, points])
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        segment_times = distances / max(float(max_velocity), 1e-6)
        segment_times = np.maximum(segment_times, sample_dt)
        times = np.concatenate([[0.0], np.cumsum(segment_times)])
        spline = CubicSpline(times, points, axis=0, bc_type="natural") if smooth and len(points) >= 3 else None
        return cls(times=times.astype(float), waypoints=points, sample_dt=float(sample_dt), spline=spline)

    def sample(self, time_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_value = float(np.clip(time_value, 0.0, self.duration))
        if self.spline is not None:
            return (
                np.asarray(self.spline(time_value, 0), dtype=float),
                np.asarray(self.spline(time_value, 1), dtype=float),
                np.asarray(self.spline(time_value, 2), dtype=float),
            )
        index = int(np.searchsorted(self.times, time_value, side="right") - 1)
        index = min(max(index, 0), len(self.times) - 2)
        start_time, end_time = self.times[index], self.times[index + 1]
        alpha = 0.0 if end_time == start_time else (time_value - start_time) / (end_time - start_time)
        start = self.waypoints[index]
        end = self.waypoints[index + 1]
        position = (1.0 - alpha) * start + alpha * end
        velocity = (end - start) / max(end_time - start_time, 1e-6)
        return position, velocity, np.zeros(3, dtype=float)

    def sample_segment(self, end_time: float, count: int = 10) -> np.ndarray:
        end_time = float(np.clip(end_time, 0.0, self.duration))
        times = np.linspace(0.0, end_time, max(int(count), 2))
        return np.stack([self.sample(time_value)[0] for time_value in times])

    def metrics(self) -> dict[str, float]:
        samples = max(int(np.ceil(self.duration / max(self.sample_dt, 1e-6))) + 1, 3)
        times = np.linspace(0.0, self.duration, samples)
        positions = np.stack([self.sample(time_value)[0] for time_value in times])
        velocities = np.stack([self.sample(time_value)[1] for time_value in times])
        accelerations = np.stack([self.sample(time_value)[2] for time_value in times])
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        acceleration_norms = np.linalg.norm(accelerations, axis=1)
        dt = max(float(times[1] - times[0]), 1e-6)
        jerks = np.diff(accelerations, axis=0) / dt
        jerk_norms = np.linalg.norm(jerks, axis=1) if len(jerks) else np.zeros(1)
        velocity_norms = np.linalg.norm(velocities, axis=1)
        return {
            "path_length": float(np.sum(segment_lengths)),
            "mean_acceleration": float(np.mean(acceleration_norms)),
            "max_acceleration": float(np.max(acceleration_norms)),
            "smoothness_cost": float(np.mean(acceleration_norms**2) + np.mean(jerk_norms**2)),
            "mean_speed": float(np.mean(velocity_norms)),
        }


@dataclass(slots=True)
class PlannerResult:
    trajectory: ContinuousTrajectory | None
    status: str


class EGOStylePlanner:
    """Python approximation of an EGO-like local trajectory optimizer."""

    def __init__(
        self,
        max_velocity: float,
        max_acceleration: float,
        safe_obs_dist: float,
        sample_dt: float = 0.5,
        optimize_iterations: int = 30,
        smooth_weight: float = 0.35,
        obstacle_weight: float = 0.8,
        seed_connectivity: int = 26,
    ) -> None:
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.safe_obs_dist = float(safe_obs_dist)
        self.sample_dt = float(sample_dt)
        self.optimize_iterations = int(optimize_iterations)
        self.smooth_weight = float(smooth_weight)
        self.obstacle_weight = float(obstacle_weight)
        self.seed_planner = AStar3D(seed_connectivity)
        self.conservative_planner = AStar3D(6)

    def plan(
        self,
        start_state: Iterable[float],
        goal: Iterable[int],
        knowledge_states: np.ndarray,
    ) -> ContinuousTrajectory | None:
        return self.plan_with_status(start_state, goal, knowledge_states).trajectory

    def plan_with_status(
        self,
        start_state: Iterable[float],
        goal: Iterable[int],
        knowledge_states: np.ndarray,
    ) -> PlannerResult:
        start_voxel = tuple(int(round(value)) for value in start_state)
        goal_voxel = tuple(int(value) for value in goal)
        seed_path = self.seed_planner.plan(start_voxel, goal_voxel, knowledge_states)
        if seed_path is None:
            return PlannerResult(None, "failed_no_seed_path")

        control_points = self._compress_path(np.asarray(seed_path, dtype=float))
        optimized_points = self._optimize_control_points(control_points, knowledge_states)
        trajectory = ContinuousTrajectory.from_waypoints(
            optimized_points,
            max_velocity=self.max_velocity,
            sample_dt=self.sample_dt,
            smooth=True,
        )
        if self._trajectory_is_valid(trajectory, knowledge_states):
            return PlannerResult(trajectory, "optimized")

        fallback = ContinuousTrajectory.from_waypoints(
            np.asarray(seed_path, dtype=float),
            max_velocity=self.max_velocity,
            sample_dt=self.sample_dt,
            smooth=False,
        )
        if self._trajectory_is_valid(fallback, knowledge_states):
            return PlannerResult(fallback, "raw_seed_fallback")

        conservative_path = self.conservative_planner.plan(start_voxel, goal_voxel, knowledge_states)
        if conservative_path is None:
            return PlannerResult(None, "failed_no_conservative_path")

        conservative_fallback = ContinuousTrajectory.from_waypoints(
            np.asarray(conservative_path, dtype=float),
            max_velocity=self.max_velocity,
            sample_dt=self.sample_dt,
            smooth=False,
        )
        if self._trajectory_is_valid(conservative_fallback, knowledge_states):
            return PlannerResult(conservative_fallback, "axis_aligned_fallback")
        return PlannerResult(None, "failed_all_fallbacks")

    def _optimize_control_points(self, points: np.ndarray, states: np.ndarray) -> np.ndarray:
        if len(points) <= 2:
            return points
        optimized = points.copy()
        traversable_mask = np.isin(states, [int(state) for state in TRAVERSABLE_STATES])
        distance_field = distance_transform_edt(traversable_mask)
        gradients = []
        for axis, axis_size in enumerate(states.shape):
            if axis_size < 2:
                gradients.append(np.zeros_like(distance_field, dtype=float))
            else:
                gradients.append(np.gradient(distance_field.astype(float), axis=axis))
        bounds = np.asarray(states.shape, dtype=float) - 1.0
        clearance_target = self.safe_obs_dist + 0.5

        for _ in range(self.optimize_iterations):
            for index in range(1, len(optimized) - 1):
                current = optimized[index]
                smooth_push = 0.5 * (optimized[index - 1] + optimized[index + 1]) - current
                voxel = self._clip_index(current, states.shape)
                distance = float(distance_field[voxel])
                obstacle_push = np.zeros(3, dtype=float)
                if distance < clearance_target:
                    gradient = np.asarray([axis[voxel] for axis in gradients], dtype=float)
                    norm = float(np.linalg.norm(gradient))
                    if norm > 1e-6:
                        obstacle_push = gradient / norm * (clearance_target - distance)
                candidate = current + self.smooth_weight * smooth_push + self.obstacle_weight * obstacle_push
                candidate = np.clip(candidate, 0.0, bounds)
                candidate_voxel = self._clip_index(candidate, states.shape)
                if traversable_mask[candidate_voxel]:
                    optimized[index] = candidate
        return optimized

    def _trajectory_is_valid(self, trajectory: ContinuousTrajectory, states: np.ndarray) -> bool:
        samples = max(int(np.ceil(trajectory.duration / max(self.sample_dt / 2.0, 1e-6))) + 1, 3)
        traversable = {int(state) for state in TRAVERSABLE_STATES}
        for time_value in np.linspace(0.0, trajectory.duration, samples):
            position, _, acceleration = trajectory.sample(float(time_value))
            if np.linalg.norm(acceleration) > self.max_acceleration * 4.0:
                return False
            voxel = self._clip_index(position, states.shape)
            if int(states[voxel]) not in traversable:
                return False
        return True

    @staticmethod
    def _compress_path(points: np.ndarray) -> np.ndarray:
        if len(points) <= 2:
            return points
        kept = [points[0]]
        previous_direction = points[1] - points[0]
        for index in range(1, len(points) - 1):
            current_direction = points[index + 1] - points[index]
            if not np.array_equal(current_direction, previous_direction):
                kept.append(points[index])
            previous_direction = current_direction
        kept.append(points[-1])
        return np.asarray(kept, dtype=float)

    @staticmethod
    def _clip_index(position: np.ndarray, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        rounded = np.rint(position).astype(int)
        clipped = np.clip(rounded, 0, np.asarray(shape) - 1)
        return tuple(int(value) for value in clipped)
