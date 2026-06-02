from __future__ import annotations

from typing import Iterable

import numpy as np

from safe_ctde_mace.mapping.coverage_map import CoverageMap
from safe_ctde_mace.mapping.voxel_map import VoxelState


class MapFusion:
    """Fuse local maps among communication neighbors."""

    @staticmethod
    def fuse_neighbors(maps: list[CoverageMap], neighbor_lists: list[list[int]]) -> None:
        snapshots = [coverage_map.clone() for coverage_map in maps]
        for index, coverage_map in enumerate(maps):
            merged_base = snapshots[index].base_states.copy()
            merged_reserved = snapshots[index].reserved_mask.copy()
            for neighbor_index in neighbor_lists[index]:
                neighbor = snapshots[neighbor_index]
                merged_base = MapFusion._merge_base_states(merged_base, neighbor.base_states)
                merged_reserved |= neighbor.reserved_mask
            coverage_map.base_states = merged_base
            coverage_map.reserved_mask = merged_reserved

    @staticmethod
    def _merge_base_states(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if first.shape != second.shape:
            raise ValueError("Voxel maps must have identical shapes.")

        result = first.copy()
        free_mask = (second == int(VoxelState.FREE)) & (result == int(VoxelState.UNKNOWN)) & (result != int(VoxelState.COVERED))
        covered_mask = (second == int(VoxelState.COVERED)) & (result != int(VoxelState.OBSTACLE))
        result[free_mask] = int(VoxelState.FREE)
        result[covered_mask] = int(VoxelState.COVERED)
        result[second == int(VoxelState.OBSTACLE)] = int(VoxelState.OBSTACLE)
        return result

    @staticmethod
    def compress_indices(indices: Iterable[Iterable[int]], shape: tuple[int, int, int]) -> list[int]:
        tuples = [tuple(int(value) for value in index) for index in indices]
        if not tuples:
            return []
        coordinates = np.asarray(tuples, dtype=int).T
        return [int(value) for value in np.ravel_multi_index(coordinates, shape)]

    @staticmethod
    def decompress_indices(flat_indices: Iterable[int], shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        values = list(flat_indices)
        if not values:
            return []
        coordinates = np.unravel_index(values, shape)
        return [tuple(int(axis_values[index]) for axis_values in coordinates) for index in range(len(values))]
