from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from safe_ctde_mace.mapping.voxel_map import VoxelState


@dataclass(slots=True)
class BoxObstacle:
    min_corner: tuple[int, int, int]
    max_corner: tuple[int, int, int]


class VoxelWorld:
    """Ground-truth 3D occupancy world."""

    def __init__(
        self,
        grid_size: Iterable[int],
        voxel_resolution: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.grid_size = tuple(int(value) for value in grid_size)
        self.voxel_resolution = float(voxel_resolution)
        self.rng = np.random.default_rng(seed)
        self.grid = np.full(self.grid_size, int(VoxelState.FREE), dtype=np.int8)

    @property
    def free_voxel_count(self) -> int:
        return int(np.count_nonzero(self.grid == int(VoxelState.FREE)))

    def reset(self) -> None:
        self.grid.fill(int(VoxelState.FREE))

    def in_bounds(self, index: Iterable[int]) -> bool:
        voxel = np.asarray(tuple(index), dtype=int)
        return bool(np.all(voxel >= 0) and np.all(voxel < np.asarray(self.grid_size)))

    def is_obstacle(self, index: Iterable[int]) -> bool:
        if not self.in_bounds(index):
            return True
        return bool(self.grid[tuple(index)] == int(VoxelState.OBSTACLE))

    def add_box(self, min_corner: Iterable[int], max_corner: Iterable[int]) -> BoxObstacle:
        min_arr = np.clip(np.asarray(tuple(min_corner), dtype=int), 0, np.asarray(self.grid_size) - 1)
        max_arr = np.clip(np.asarray(tuple(max_corner), dtype=int), 0, np.asarray(self.grid_size) - 1)
        if np.any(max_arr < min_arr):
            raise ValueError("max_corner must be greater than or equal to min_corner.")

        slices = tuple(slice(int(start), int(stop) + 1) for start, stop in zip(min_arr, max_arr, strict=True))
        self.grid[slices] = int(VoxelState.OBSTACLE)
        return BoxObstacle(tuple(min_arr.tolist()), tuple(max_arr.tolist()))

    def add_random_obstacles(
        self,
        count: int,
        min_box_size: Iterable[int],
        max_box_size: Iterable[int],
        forbidden_positions: Iterable[Iterable[int]] | None = None,
    ) -> list[BoxObstacle]:
        min_size = np.asarray(tuple(min_box_size), dtype=int)
        max_size = np.asarray(tuple(max_box_size), dtype=int)
        forbidden = {tuple(int(value) for value in position) for position in (forbidden_positions or [])}
        boxes: list[BoxObstacle] = []

        attempts = 0
        while len(boxes) < count and attempts < count * 20:
            attempts += 1
            size = self.rng.integers(min_size, max_size + 1)
            max_start = np.maximum(np.asarray(self.grid_size) - size, 0)
            start = np.array(
                [self.rng.integers(0, int(limit) + 1) for limit in max_start],
                dtype=int,
            )
            stop = start + size - 1
            candidate_positions = {
                tuple(index)
                for index in np.ndindex(tuple((stop - start + 1).tolist()))
            }
            candidate_positions = {
                tuple((np.asarray(index, dtype=int) + start).tolist())
                for index in candidate_positions
            }
            if forbidden & candidate_positions:
                continue
            boxes.append(self.add_box(start, stop))
        return boxes

    def get_local_patch(self, center: Iterable[int], radius: int) -> np.ndarray:
        center_arr = np.asarray(tuple(center), dtype=int)
        width = 2 * radius + 1
        patch = np.full((width, width, width), int(VoxelState.OBSTACLE), dtype=np.int8)
        for local_index in np.ndindex(patch.shape):
            offset = np.asarray(local_index, dtype=int) - radius
            world_index = center_arr + offset
            if self.in_bounds(world_index):
                patch[local_index] = self.grid[tuple(world_index)]
        return patch

    def coverage_ratio(self, coverage_states: np.ndarray) -> float:
        covered = np.count_nonzero(coverage_states == int(VoxelState.COVERED))
        return float(covered / max(self.free_voxel_count, 1))

    def visualize(self, ax=None):
        from safe_ctde_mace.utils.visualization import plot_world

        return plot_world(self, ax=ax)

