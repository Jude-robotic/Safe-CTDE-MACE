from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np

from safe_ctde_mace.mapping.voxel_map import TRAVERSABLE_STATES
from safe_ctde_mace.utils.geometry import euclidean_distance, neighbor_offsets


class AStar3D:
    """Voxel-grid A* planner."""

    def __init__(self, connectivity: int = 6) -> None:
        self.connectivity = connectivity

    def plan(
        self,
        start: Iterable[int],
        goal: Iterable[int],
        states: np.ndarray,
    ) -> list[tuple[int, int, int]] | None:
        start_voxel = tuple(int(value) for value in start)
        goal_voxel = tuple(int(value) for value in goal)
        if not self._is_traversable(start_voxel, states) or not self._is_traversable(goal_voxel, states):
            return None

        frontier: list[tuple[float, tuple[int, int, int]]] = [(0.0, start_voxel)]
        came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_voxel: None}
        g_cost = {start_voxel: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_voxel:
                return self._reconstruct_path(came_from, current)

            for offset in neighbor_offsets(self.connectivity):
                neighbor = tuple(
                    int(value + delta)
                    for value, delta in zip(current, offset, strict=True)
                )
                if not self._is_traversable(neighbor, states):
                    continue

                step_cost = euclidean_distance(current, neighbor)
                candidate_cost = g_cost[current] + step_cost
                if candidate_cost >= g_cost.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_cost[neighbor] = candidate_cost
                priority = candidate_cost + euclidean_distance(neighbor, goal_voxel)
                heapq.heappush(frontier, (priority, neighbor))
        return None

    @staticmethod
    def _reconstruct_path(
        came_from: dict[tuple[int, int, int], tuple[int, int, int] | None],
        current: tuple[int, int, int],
    ) -> list[tuple[int, int, int]]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]  # type: ignore[assignment]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _is_traversable(index: tuple[int, int, int], states: np.ndarray) -> bool:
        if any(value < 0 or value >= limit for value, limit in zip(index, states.shape, strict=True)):
            return False
        return int(states[index]) in {int(state) for state in TRAVERSABLE_STATES}

