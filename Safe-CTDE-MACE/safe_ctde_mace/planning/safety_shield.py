from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import distance_transform_edt

from safe_ctde_mace.mapping.voxel_map import TRAVERSABLE_STATES, VoxelState
from safe_ctde_mace.utils.geometry import euclidean_distance, spherical_offsets


@dataclass(slots=True)
class ShieldResult:
    safe_goal: tuple[int, int, int]
    path: list[tuple[int, int, int]]
    chosen_index: int | None
    status: str
    hover_reason: str | None = None


class SafetyShield:
    """Engineering safety layer that screens high-level RL goals."""

    def __init__(self, safe_obs_dist: float, safe_agent_dist: float) -> None:
        self.safe_obs_dist = float(safe_obs_dist)
        self.safe_agent_dist = float(safe_agent_dist)

    def select_safe_goal(
        self,
        current_position: Iterable[int],
        candidate_goals: np.ndarray,
        action_mask: np.ndarray,
        chosen_action: int,
        knowledge_states: np.ndarray,
        neighbor_states: Iterable[dict[str, np.ndarray]],
        planner,
        obstacle_distance: np.ndarray | None = None,
    ) -> ShieldResult:
        current = tuple(int(value) for value in current_position)
        order = self._candidate_order(action_mask, chosen_action)
        if not order:
            return ShieldResult(current, [current], None, "hover", "no_valid_candidate")
        obstacle_distance = (
            self._obstacle_distance_field(knowledge_states)
            if obstacle_distance is None
            else obstacle_distance
        )
        saw_neighbor_conflict = False
        saw_planner_unavailable = False

        for candidate_index in order:
            goal = tuple(int(value) for value in candidate_goals[candidate_index])
            if not self._in_bounds(goal, knowledge_states.shape):
                continue
            adjusted_goal = self._safe_or_adjusted_goal(goal, knowledge_states, obstacle_distance)
            if adjusted_goal is None:
                continue
            if self._too_close_to_neighbors(adjusted_goal, neighbor_states):
                saw_neighbor_conflict = True
                continue
            path = planner.plan(current, adjusted_goal, knowledge_states)
            if path is not None:
                status = "adjusted" if adjusted_goal != goal else "safe"
                return ShieldResult(adjusted_goal, path, int(candidate_index), status)
            saw_planner_unavailable = True

        if saw_neighbor_conflict:
            hover_reason = "neighbor_conflict"
        elif saw_planner_unavailable:
            hover_reason = "planner_unavailable"
        else:
            hover_reason = "shield_rejected"
        return ShieldResult(current, [current], None, "hover", hover_reason)

    def cbf_qp(self, *args, **kwargs):
        raise NotImplementedError("CBF-QP is reserved for the second implementation stage.")

    def _safe_or_adjusted_goal(
        self,
        goal: tuple[int, int, int],
        states: np.ndarray,
        obstacle_distance: np.ndarray,
    ) -> tuple[int, int, int] | None:
        if states[goal] == int(VoxelState.OBSTACLE):
            return None
        if int(states[goal]) not in {int(state) for state in TRAVERSABLE_STATES}:
            return None
        if obstacle_distance[goal] >= self.safe_obs_dist:
            return goal

        sorted_offsets = sorted(spherical_offsets(max(self.safe_obs_dist * 2.0, 1.0)), key=np.linalg.norm)
        for offset in sorted_offsets:
            candidate = tuple(int(value + delta) for value, delta in zip(goal, offset, strict=True))
            if not self._in_bounds(candidate, states.shape):
                continue
            if int(states[candidate]) not in {int(state) for state in TRAVERSABLE_STATES}:
                continue
            if obstacle_distance[candidate] >= self.safe_obs_dist:
                return candidate
        return None

    def _too_close_to_neighbors(
        self,
        goal: tuple[int, int, int],
        neighbor_states: Iterable[dict[str, np.ndarray]],
    ) -> bool:
        for neighbor in neighbor_states:
            predicted = np.asarray(neighbor["position"], dtype=float) + np.asarray(
                neighbor.get("velocity", np.zeros(3)),
                dtype=float,
            )
            if euclidean_distance(goal, predicted) < self.safe_agent_dist:
                return True
        return False

    @staticmethod
    def _candidate_order(action_mask: np.ndarray, chosen_action: int) -> list[int]:
        valid = [int(index) for index in np.flatnonzero(action_mask)]
        if chosen_action in valid:
            return [chosen_action] + [index for index in valid if index != chosen_action]
        return valid

    @staticmethod
    def _obstacle_distance_field(states: np.ndarray) -> np.ndarray:
        obstacle_mask = states == int(VoxelState.OBSTACLE)
        return distance_transform_edt(~obstacle_mask)

    @staticmethod
    def _in_bounds(index: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
        return all(0 <= value < limit for value, limit in zip(index, shape, strict=True))
