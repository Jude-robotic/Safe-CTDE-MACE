from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safe_ctde_mace.mapping.voxel_map import VoxelState


@dataclass(slots=True)
class EpisodeFrame:
    step: int
    coverage_ratio: float
    covered_mask: np.ndarray
    frontier_mask: np.ndarray
    uav_positions: np.ndarray


@dataclass(slots=True)
class EpisodeReplay:
    grid_size: tuple[int, int, int]
    obstacle_mask: np.ndarray
    planner_type: str
    frames: list[EpisodeFrame] = field(default_factory=list)


def start_episode_replay(env) -> EpisodeReplay:
    """Create a replay container and capture the reset state."""
    if env.world is None:
        raise RuntimeError("Environment must be reset before capturing a replay.")
    replay = EpisodeReplay(
        grid_size=tuple(int(value) for value in env.world.grid_size),
        obstacle_mask=(env.world.grid == int(VoxelState.OBSTACLE)).copy(),
        planner_type=str(env.motion_planner_type),
    )
    replay.frames.append(capture_episode_frame(env))
    return replay


def capture_episode_frame(env) -> EpisodeFrame:
    """Snapshot the visible state needed for static or animated replay."""
    if env.world is None or env.global_coverage is None:
        raise RuntimeError("Environment must be reset before capturing a replay frame.")

    frontier_mask = np.zeros(env.world.grid_size, dtype=bool)
    frontier_points = {voxel for frontier_set in env.frontier_sets for voxel in frontier_set}
    if frontier_points:
        frontier_indices = tuple(np.asarray(sorted(frontier_points), dtype=int).T)
        frontier_mask[frontier_indices] = True

    return EpisodeFrame(
        step=int(env.step_count),
        coverage_ratio=float(env.global_coverage.coverage_ratio(env.world)),
        covered_mask=(env.global_coverage.base_states == int(VoxelState.COVERED)).copy(),
        frontier_mask=frontier_mask,
        uav_positions=np.stack([agent.position.copy() for agent in env.agents]).astype(float),
    )
