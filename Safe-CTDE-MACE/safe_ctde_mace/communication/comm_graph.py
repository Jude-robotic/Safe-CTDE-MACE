from __future__ import annotations

from typing import Iterable

import numpy as np

from safe_ctde_mace.utils.geometry import euclidean_distance


class CommGraph:
    """Distance-based communication graph."""

    def __init__(self, comm_range: float, sigma_c: float | None = None) -> None:
        self.comm_range = float(comm_range)
        self.sigma_c = sigma_c

    def adjacency_matrix(self, positions: Iterable[Iterable[float]]) -> np.ndarray:
        positions_list = [np.asarray(position, dtype=float) for position in positions]
        count = len(positions_list)
        adjacency = np.zeros((count, count), dtype=bool)
        for source in range(count):
            for target in range(source + 1, count):
                connected = euclidean_distance(positions_list[source], positions_list[target]) <= self.comm_range
                adjacency[source, target] = connected
                adjacency[target, source] = connected
        return adjacency

    def neighbor_lists(self, positions: Iterable[Iterable[float]]) -> list[list[int]]:
        adjacency = self.adjacency_matrix(positions)
        return [list(np.flatnonzero(row)) for row in adjacency]

    def communication_probability(self, distance: float) -> float:
        if self.sigma_c is None:
            return 1.0 if distance <= self.comm_range else 0.0
        return float(np.exp(-(distance**2) / (2.0 * self.sigma_c**2)))

