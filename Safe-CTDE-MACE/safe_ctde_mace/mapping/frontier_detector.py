from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable

import numpy as np

from safe_ctde_mace.mapping.voxel_map import TRAVERSABLE_STATES, VoxelState
from safe_ctde_mace.utils.geometry import euclidean_distance, neighbor_offsets, spherical_offsets_array

CANDIDATE_FEATURE_SCHEMA_VERSION = 2


@dataclass(slots=True)
class CandidateSet:
    goals: np.ndarray
    features: np.ndarray
    action_mask: np.ndarray
    frontier_voxels: set[tuple[int, int, int]]


@dataclass(frozen=True, slots=True)
class CandidateFeatureLayout:
    """Stable offsets for variable-width candidate feature vectors."""

    max_neighbors: int

    @property
    def assignment_margin_slice(self) -> slice:
        return slice(6, 6 + self.max_neighbors)

    @property
    def grid_quadrant(self) -> int:
        return 6 + self.max_neighbors

    @property
    def layer_height(self) -> int:
        return 7 + self.max_neighbors

    @property
    def uncovered_density(self) -> int:
        return 8 + self.max_neighbors

    @property
    def size(self) -> int:
        return 9 + self.max_neighbors

    @classmethod
    def from_feature_width(cls, width: int) -> "CandidateFeatureLayout":
        return cls(max_neighbors=max(int(width) - 9, 0))


def candidate_score(features: np.ndarray, layout: CandidateFeatureLayout) -> float:
    distance_to_uav, info_gain, obstacle_risk, reserved_penalty, neighbor_overlap, path_cost = features[:6]
    uncovered_density = features[layout.uncovered_density]
    return float(
        1.5 * info_gain
        - 0.2 * distance_to_uav
        - 2.0 * obstacle_risk
        - 1.5 * reserved_penalty
        - 1.0 * neighbor_overlap
        - 0.2 * path_cost
        + 0.3 * uncovered_density
    )


class FrontierDetector:
    """Generate fixed-size candidate frontier goal sets."""

    def __init__(
        self,
        num_candidates: int,
        sensor_range: float,
        obstacle_radius: float = 2.0,
        reservation_radius: float = 1.5,
        candidate_min_separation: float = 0.0,
        max_neighbors: int = 0,
    ) -> None:
        self.num_candidates = int(num_candidates)
        self.sensor_range = float(sensor_range)
        self.obstacle_radius = float(obstacle_radius)
        self.reservation_radius = float(reservation_radius)
        self.candidate_min_separation = float(candidate_min_separation)
        self.max_neighbors = int(max_neighbors)
        self.feature_layout = CandidateFeatureLayout(self.max_neighbors)
        self.sensor_window_size = len(spherical_offsets_array(self.sensor_range))

    def detect_frontiers(self, states: np.ndarray) -> set[tuple[int, int, int]]:
        traversable_values = {int(state) for state in TRAVERSABLE_STATES}
        traversable = np.isin(states, list(traversable_values))
        unknown = states == int(VoxelState.UNKNOWN)
        adjacent_unknown = np.zeros_like(unknown, dtype=bool)
        adjacent_unknown[1:, :, :] |= unknown[:-1, :, :]
        adjacent_unknown[:-1, :, :] |= unknown[1:, :, :]
        adjacent_unknown[:, 1:, :] |= unknown[:, :-1, :]
        adjacent_unknown[:, :-1, :] |= unknown[:, 1:, :]
        adjacent_unknown[:, :, 1:] |= unknown[:, :, :-1]
        adjacent_unknown[:, :, :-1] |= unknown[:, :, 1:]
        return {
            tuple(int(value) for value in index)
            for index in np.argwhere(traversable & adjacent_unknown)
        }

    def cluster_frontiers(self, frontiers: set[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
        remaining = set(frontiers)
        clusters: list[list[tuple[int, int, int]]] = []
        while remaining:
            seed = remaining.pop()
            cluster = [seed]
            queue = [seed]
            while queue:
                current = queue.pop()
                for offset in neighbor_offsets(6):
                    neighbor = tuple(
                        int(value + delta)
                        for value, delta in zip(current, offset, strict=True)
                    )
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
                        cluster.append(neighbor)
            clusters.append(cluster)
        return clusters

    def generate_candidates(
        self,
        states: np.ndarray,
        uav_position: Iterable[int],
        neighbor_positions: Iterable[Iterable[float]] | None = None,
    ) -> CandidateSet:
        reachable_distances = self._reachable_distances(states, tuple(int(value) for value in uav_position))
        frontiers = self.detect_frontiers(states) & set(reachable_distances)
        clusters = self.cluster_frontiers(frontiers)
        neighbor_positions = list(neighbor_positions or [])
        scored: list[tuple[float, tuple[int, int, int], np.ndarray]] = []

        for cluster in clusters:
            representative = self._representative(cluster)
            features = self.compute_features(
                states,
                representative,
                uav_position,
                neighbor_positions,
                path_cost=reachable_distances[representative],
            )
            score = self._score(features)
            scored.append((score, representative, features))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = self._select_diverse_candidates(scored)

        goals = np.full((self.num_candidates, 3), -1, dtype=np.int16)
        features = np.zeros((self.num_candidates, self.feature_layout.size), dtype=np.float32)
        action_mask = np.zeros(self.num_candidates, dtype=bool)
        for index, (_, goal, feature_vector) in enumerate(selected):
            goals[index] = np.asarray(goal, dtype=np.int16)
            features[index] = feature_vector
            action_mask[index] = True

        return CandidateSet(goals, features, action_mask, frontiers)

    def _select_diverse_candidates(
        self,
        scored: list[tuple[float, tuple[int, int, int], np.ndarray]],
    ) -> list[tuple[float, tuple[int, int, int], np.ndarray]]:
        if self.candidate_min_separation <= 0.0:
            return scored[: self.num_candidates]

        selected: list[tuple[float, tuple[int, int, int], np.ndarray]] = []
        deferred: list[tuple[float, tuple[int, int, int], np.ndarray]] = []
        for item in scored:
            _, goal, _ = item
            if all(
                euclidean_distance(goal, selected_goal) >= self.candidate_min_separation
                for _, selected_goal, _ in selected
            ):
                selected.append(item)
            else:
                deferred.append(item)
            if len(selected) == self.num_candidates:
                return selected

        selected.extend(deferred[: max(self.num_candidates - len(selected), 0)])
        return selected[: self.num_candidates]

    def compute_features(
        self,
        states: np.ndarray,
        candidate: tuple[int, int, int],
        uav_position: Iterable[int],
        neighbor_positions: Iterable[Iterable[float]],
        path_cost: float | None = None,
    ) -> np.ndarray:
        distance_to_uav = euclidean_distance(candidate, uav_position)
        expected_information_gain = self._expected_information_gain(states, candidate)
        obstacle_risk = self._density(states, candidate, self.obstacle_radius, VoxelState.OBSTACLE)
        reserved_penalty = self._density(states, candidate, self.reservation_radius, VoxelState.RESERVED)
        neighbor_overlap = self._neighbor_overlap(candidate, neighbor_positions)
        path_cost = distance_to_uav if path_cost is None else float(path_cost)
        assignment_margins = self._assignment_margins(candidate, neighbor_positions, path_cost)
        grid_quadrant = self._grid_quadrant(candidate, states.shape)
        layer_height = self._layer_height(candidate, states.shape)
        uncovered_density = self._uncovered_density(states, candidate)
        return np.asarray(
            [
                distance_to_uav,
                expected_information_gain,
                obstacle_risk,
                reserved_penalty,
                neighbor_overlap,
                path_cost,
                *assignment_margins,
                grid_quadrant,
                layer_height,
                uncovered_density,
            ],
            dtype=np.float32,
        )

    def _expected_information_gain(self, states: np.ndarray, candidate: tuple[int, int, int]) -> float:
        values = self._window_values(states, candidate, self.sensor_range)
        if len(values) == 0:
            return 0.0
        unknown_ratio = np.count_nonzero(values == int(VoxelState.UNKNOWN)) / len(values)
        return float(unknown_ratio * self.sensor_window_size)

    def _density(
        self,
        states: np.ndarray,
        candidate: tuple[int, int, int],
        radius: float,
        target_state: VoxelState,
    ) -> float:
        values = self._window_values(states, candidate, radius)
        return float(np.count_nonzero(values == int(target_state)) / len(values)) if len(values) else 0.0

    def _neighbor_overlap(
        self,
        candidate: tuple[int, int, int],
        neighbor_positions: Iterable[Iterable[float]],
    ) -> float:
        overlap = 0.0
        for neighbor_position in neighbor_positions:
            distance = euclidean_distance(candidate, neighbor_position)
            overlap += max(0.0, 1.0 - distance / max(2.0 * self.sensor_range, 1e-6))
        return float(overlap)

    def _assignment_margins(
        self,
        candidate: tuple[int, int, int],
        neighbor_positions: Iterable[Iterable[float]],
        path_cost: float,
    ) -> list[float]:
        margins = [
            euclidean_distance(candidate, neighbor_position) - path_cost
            for neighbor_position in list(neighbor_positions)[: self.max_neighbors]
        ]
        return [*margins, *([0.0] * (self.max_neighbors - len(margins)))]

    def _reachable_distances(
        self,
        states: np.ndarray,
        start: tuple[int, int, int],
    ) -> dict[tuple[int, int, int], float]:
        traversable_values = {int(state) for state in TRAVERSABLE_STATES}
        if not self._in_bounds(start, states.shape) or int(states[start]) not in traversable_values:
            return {}

        distances = {start: 0.0}
        queue: deque[tuple[int, int, int]] = deque([start])
        while queue:
            current = queue.popleft()
            for offset in neighbor_offsets(6):
                neighbor = tuple(
                    int(value + delta)
                    for value, delta in zip(current, offset, strict=True)
                )
                if not self._in_bounds(neighbor, states.shape):
                    continue
                if int(states[neighbor]) not in traversable_values or neighbor in distances:
                    continue
                distances[neighbor] = distances[current] + 1.0
                queue.append(neighbor)
        return distances

    @staticmethod
    def _representative(cluster: list[tuple[int, int, int]]) -> tuple[int, int, int]:
        centroid = np.mean(np.asarray(cluster, dtype=float), axis=0)
        return min(cluster, key=lambda voxel: euclidean_distance(voxel, centroid))

    def _score(self, features: np.ndarray) -> float:
        return candidate_score(features, self.feature_layout)

    @staticmethod
    def _in_bounds(index: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
        return all(0 <= value < limit for value, limit in zip(index, shape, strict=True))

    @staticmethod
    def _window_values(states: np.ndarray, candidate: tuple[int, int, int], radius: float) -> np.ndarray:
        offsets = spherical_offsets_array(radius)
        voxels = offsets + np.asarray(candidate, dtype=int)
        valid = np.all((voxels >= 0) & (voxels < np.asarray(states.shape)), axis=1)
        valid_voxels = voxels[valid]
        if len(valid_voxels) == 0:
            return np.asarray([], dtype=states.dtype)
        return states[tuple(valid_voxels.T)]

    def _grid_quadrant(self, candidate: tuple[int, int, int], grid_shape: tuple[int, int, int]) -> float:
        x, y, _ = candidate
        x_mid = grid_shape[0] / 2.0
        y_mid = grid_shape[1] / 2.0
        quadrant = (0 if x < x_mid else 2) + (1 if y < y_mid else 0)
        return float(quadrant) / 3.0

    def _layer_height(self, candidate: tuple[int, int, int], grid_shape: tuple[int, int, int]) -> float:
        z = candidate[2]
        max_z = max(grid_shape[2] - 1, 1)
        return float(z) / float(max_z)

    def _uncovered_density(self, states: np.ndarray, candidate: tuple[int, int, int]) -> float:
        values = self._window_values(states, candidate, 3.0)
        if len(values) == 0:
            return 0.0
        unknown_count = np.count_nonzero(values == int(VoxelState.UNKNOWN))
        return float(unknown_count) / len(values)
