from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Iterable

import numpy as np


def euclidean_distance(first: Iterable[float], second: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


@lru_cache(maxsize=None)
def spherical_offsets(radius: float) -> tuple[tuple[int, int, int], ...]:
    """Return integer voxel offsets within a Euclidean sphere."""
    ceil_radius = int(np.ceil(radius))
    offsets: list[tuple[int, int, int]] = []
    for dx, dy, dz in product(range(-ceil_radius, ceil_radius + 1), repeat=3):
        if dx * dx + dy * dy + dz * dz <= radius * radius + 1e-9:
            offsets.append((dx, dy, dz))
    return tuple(offsets)


@lru_cache(maxsize=None)
def spherical_offsets_array(radius: float) -> np.ndarray:
    """Return cached spherical offsets as an immutable NumPy array."""
    offsets = np.asarray(spherical_offsets(radius), dtype=int)
    offsets.setflags(write=False)
    return offsets


@lru_cache(maxsize=None)
def neighbor_offsets(connectivity: int = 6) -> tuple[tuple[int, int, int], ...]:
    if connectivity == 6:
        return (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        )
    if connectivity == 26:
        return tuple(
            (dx, dy, dz)
            for dx, dy, dz in product((-1, 0, 1), repeat=3)
            if not (dx == dy == dz == 0)
        )
    raise ValueError("Only 6- and 26-neighborhoods are supported.")


def clip_voxel(index: Iterable[int], shape: tuple[int, int, int]) -> tuple[int, int, int]:
    arr = np.asarray(index, dtype=int)
    clipped = np.clip(arr, 0, np.asarray(shape, dtype=int) - 1)
    return tuple(int(value) for value in clipped)
