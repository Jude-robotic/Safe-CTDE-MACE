from __future__ import annotations

from enum import IntEnum

import numpy as np


class VoxelState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    COVERED = 3
    RESERVED = 4


PRIORITY_ORDER = {
    VoxelState.UNKNOWN: 0,
    VoxelState.FREE: 1,
    VoxelState.RESERVED: 2,
    VoxelState.COVERED: 3,
    VoxelState.OBSTACLE: 4,
}

TRAVERSABLE_STATES = {
    VoxelState.FREE,
    VoxelState.COVERED,
    VoxelState.RESERVED,
}


def merge_state_arrays(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Merge two voxel-state arrays using the project-wide priority order."""
    if first.shape != second.shape:
        raise ValueError("Voxel maps must have the same shape to merge.")

    result = first.copy()
    for state in sorted(PRIORITY_ORDER, key=PRIORITY_ORDER.get):
        mask = second == int(state)
        higher_priority = PRIORITY_ORDER[state] > np.vectorize(
            lambda value: PRIORITY_ORDER[VoxelState(int(value))]
        )(result)
        result[mask & higher_priority] = int(state)
    return result


def encode_state_channels(states: np.ndarray) -> np.ndarray:
    """Return a 5-channel one-hot representation of voxel states."""
    return np.stack([(states == int(state)).astype(np.float32) for state in VoxelState], axis=0)

