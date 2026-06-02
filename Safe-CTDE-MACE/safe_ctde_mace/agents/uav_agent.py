from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from safe_ctde_mace.mapping.coverage_map import CoverageMap, SensorUpdate
from safe_ctde_mace.utils.geometry import spherical_offsets


@dataclass(slots=True)
class UAVAgent:
    agent_id: int
    position: np.ndarray
    sensor_range: float
    local_map: CoverageMap
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    goal: np.ndarray | None = None
    reserved_region: set[tuple[int, int, int]] = field(default_factory=set)
    active: bool = True
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        agent_id: int,
        position: Iterable[int],
        sensor_range: float,
        map_shape: tuple[int, int, int],
    ) -> "UAVAgent":
        initial_position = np.asarray(tuple(position), dtype=float)
        agent = cls(agent_id, initial_position, sensor_range, CoverageMap(map_shape))
        agent.trajectory.append(tuple(float(value) for value in initial_position))
        return agent

    @property
    def current_voxel(self) -> tuple[int, int, int]:
        return tuple(int(round(value)) for value in self.position)

    def observe(self, world) -> SensorUpdate:
        return self.local_map.update_from_sensor(world, self.current_voxel, self.sensor_range)

    def set_goal(self, goal: Iterable[int] | None) -> None:
        self.goal = None if goal is None else np.asarray(tuple(goal), dtype=float)

    def reserve_region_around(self, goal: Iterable[int], radius: float) -> set[tuple[int, int, int]]:
        goal_arr = np.asarray(tuple(goal), dtype=int)
        region = {
            tuple(int(value) for value in goal_arr + np.asarray(offset, dtype=int))
            for offset in spherical_offsets(radius)
            if self.local_map.in_bounds(goal_arr + np.asarray(offset, dtype=int))
        }
        self.reserved_region = region
        self.local_map.reserve(region)
        return region

    def clear_reservation(self) -> None:
        self.reserved_region.clear()

    def update_motion(self, position: Iterable[float], velocity: Iterable[float]) -> None:
        self.position = np.asarray(tuple(position), dtype=float)
        self.velocity = np.asarray(tuple(velocity), dtype=float)
        self.trajectory.append(tuple(float(value) for value in self.position))
