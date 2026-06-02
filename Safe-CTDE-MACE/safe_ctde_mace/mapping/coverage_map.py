from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from safe_ctde_mace.mapping.voxel_map import VoxelState
from safe_ctde_mace.utils.geometry import spherical_offsets


@dataclass(slots=True)
class SensorUpdate:
    new_covered: int
    repeated_covered: int
    unknown_reduction: int
    observed_indices: set[tuple[int, int, int]]


class CoverageMap:
    """Knowledge map with short-lived reservation overlays."""

    def __init__(self, shape: Iterable[int]) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.base_states = np.full(self.shape, int(VoxelState.UNKNOWN), dtype=np.int8)
        self.reserved_mask = np.zeros(self.shape, dtype=bool)

    def clone(self) -> "CoverageMap":
        cloned = CoverageMap(self.shape)
        cloned.base_states = self.base_states.copy()
        cloned.reserved_mask = self.reserved_mask.copy()
        return cloned

    def as_array(self) -> np.ndarray:
        states = self.base_states.copy()
        reservable = np.isin(
            states,
            [int(VoxelState.UNKNOWN), int(VoxelState.FREE)],
        )
        states[self.reserved_mask & reservable] = int(VoxelState.RESERVED)
        return states

    def in_bounds(self, index: Iterable[int]) -> bool:
        voxel = np.asarray(tuple(index), dtype=int)
        return bool(np.all(voxel >= 0) and np.all(voxel < np.asarray(self.shape)))

    def get_state(self, index: Iterable[int]) -> VoxelState:
        voxel = tuple(int(value) for value in index)
        return VoxelState(int(self.as_array()[voxel]))

    def mark_free(self, indices: Iterable[Iterable[int]]) -> None:
        for index in indices:
            voxel = tuple(int(value) for value in index)
            if self.base_states[voxel] == int(VoxelState.UNKNOWN):
                self.base_states[voxel] = int(VoxelState.FREE)

    def mark_obstacle(self, indices: Iterable[Iterable[int]]) -> None:
        for index in indices:
            self.base_states[tuple(int(value) for value in index)] = int(VoxelState.OBSTACLE)

    def mark_covered(self, indices: Iterable[Iterable[int]]) -> None:
        for index in indices:
            voxel = tuple(int(value) for value in index)
            if self.base_states[voxel] != int(VoxelState.OBSTACLE):
                self.base_states[voxel] = int(VoxelState.COVERED)

    def reserve(self, indices: Iterable[Iterable[int]]) -> None:
        for index in indices:
            voxel = tuple(int(value) for value in index)
            if self.in_bounds(voxel):
                self.reserved_mask[voxel] = True

    def clear_reserved(self) -> None:
        self.reserved_mask.fill(False)

    def update_from_sensor(
        self,
        world,
        position: Iterable[int],
        sensor_range: float,
    ) -> SensorUpdate:
        position_arr = np.asarray(tuple(position), dtype=int)
        new_covered = 0
        repeated_covered = 0
        unknown_reduction = 0
        observed: set[tuple[int, int, int]] = set()

        for offset in spherical_offsets(sensor_range):
            voxel_arr = position_arr + np.asarray(offset, dtype=int)
            voxel = tuple(int(value) for value in voxel_arr)
            if not world.in_bounds(voxel):
                continue

            observed.add(voxel)
            previous = VoxelState(int(self.base_states[voxel]))
            if previous == VoxelState.UNKNOWN:
                unknown_reduction += 1

            if world.is_obstacle(voxel):
                self.base_states[voxel] = int(VoxelState.OBSTACLE)
                continue

            if previous == VoxelState.COVERED:
                repeated_covered += 1
            else:
                new_covered += 1
            self.base_states[voxel] = int(VoxelState.COVERED)

        return SensorUpdate(new_covered, repeated_covered, unknown_reduction, observed)

    def coverage_ratio(self, world) -> float:
        covered = np.count_nonzero(self.base_states == int(VoxelState.COVERED))
        return float(covered / max(world.free_voxel_count, 1))

    def get_patch(
        self,
        center: Iterable[int],
        radius: int,
        states: np.ndarray | None = None,
    ) -> np.ndarray:
        center_arr = np.asarray(tuple(center), dtype=int)
        width = 2 * radius + 1
        patch = np.full((width, width, width), int(VoxelState.UNKNOWN), dtype=np.int8)
        source = self.as_array() if states is None else states
        world_start = np.maximum(center_arr - radius, 0)
        world_stop = np.minimum(center_arr + radius + 1, np.asarray(self.shape))
        patch_start = world_start - (center_arr - radius)
        patch_stop = patch_start + (world_stop - world_start)
        patch[
            patch_start[0] : patch_stop[0],
            patch_start[1] : patch_stop[1],
            patch_start[2] : patch_stop[2],
        ] = source[
            world_start[0] : world_stop[0],
            world_start[1] : world_stop[1],
            world_start[2] : world_stop[2],
        ]
        return patch

    def covered_indices(self) -> set[tuple[int, int, int]]:
        return {tuple(index) for index in np.argwhere(self.base_states == int(VoxelState.COVERED))}

    def obstacle_indices(self) -> set[tuple[int, int, int]]:
        return {tuple(index) for index in np.argwhere(self.base_states == int(VoxelState.OBSTACLE))}

    def reserved_indices(self) -> set[tuple[int, int, int]]:
        return {tuple(index) for index in np.argwhere(self.reserved_mask)}
