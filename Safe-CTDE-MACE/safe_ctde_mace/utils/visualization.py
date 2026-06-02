from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from safe_ctde_mace.mapping.voxel_map import VoxelState
from safe_ctde_mace.utils.replay import EpisodeReplay


OBSTACLE_COLOR = "#374151"
OBSTACLE_EDGE = "#111827"
COVERED_COLOR = "#93c5fd"
COVERED_EDGE = "#2563eb"
FRONTIER_COLOR = "#f97316"
FRONTIER_EDGE = "#9a3412"
UAV_COLORS = ["#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777"]


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    """Return only the exposed shell voxels of a filled boolean mask."""
    if not np.any(mask):
        return mask.copy()

    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    interior = center.copy()
    interior &= padded[:-2, 1:-1, 1:-1]
    interior &= padded[2:, 1:-1, 1:-1]
    interior &= padded[1:-1, :-2, 1:-1]
    interior &= padded[1:-1, 2:, 1:-1]
    interior &= padded[1:-1, 1:-1, :-2]
    interior &= padded[1:-1, 1:-1, 2:]
    return center & ~interior


def _display_points(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) + 0.5


def _style_axes(ax, grid_size: tuple[int, int, int]) -> None:
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_zlim(0, grid_size[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect(grid_size)
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-58)
    ax.grid(alpha=0.2)


def _draw_scene(
    ax,
    grid_size: tuple[int, int, int],
    obstacle_mask: np.ndarray,
    covered_mask: np.ndarray,
    frontier_mask: np.ndarray,
    trajectories: Sequence[np.ndarray],
    positions: np.ndarray,
    coverage_ratio: float,
    *,
    title_suffix: str = "",
) -> None:
    _style_axes(ax, grid_size)

    if np.any(obstacle_mask):
        ax.voxels(
            obstacle_mask,
            facecolors=OBSTACLE_COLOR,
            edgecolor=OBSTACLE_EDGE,
            linewidth=0.35,
            alpha=0.96,
        )

    covered_shell = _surface_mask(covered_mask)
    if np.any(covered_shell):
        ax.voxels(
            covered_shell,
            facecolors=COVERED_COLOR,
            edgecolor=COVERED_EDGE,
            linewidth=0.22,
            alpha=0.2,
        )

    frontier_points = np.argwhere(frontier_mask)
    if len(frontier_points):
        displayed_frontiers = _display_points(frontier_points)
        ax.scatter(
            displayed_frontiers[:, 0],
            displayed_frontiers[:, 1],
            displayed_frontiers[:, 2],
            color=FRONTIER_COLOR,
            edgecolors=FRONTIER_EDGE,
            linewidths=0.7,
            marker="P",
            s=38,
            depthshade=False,
        )

    for index, trajectory in enumerate(trajectories):
        if len(trajectory) == 0:
            continue
        color = UAV_COLORS[index % len(UAV_COLORS)]
        displayed_trajectory = _display_points(trajectory)
        ax.plot(
            displayed_trajectory[:, 0],
            displayed_trajectory[:, 1],
            displayed_trajectory[:, 2],
            color=color,
            linewidth=2.8,
            label=f"uav-{index}",
        )
        ax.scatter(
            [displayed_trajectory[0, 0]],
            [displayed_trajectory[0, 1]],
            [displayed_trajectory[0, 2]],
            facecolors="white",
            edgecolors=color,
            linewidths=1.6,
            s=44,
            depthshade=False,
        )
        displayed_position = _display_points(np.asarray([positions[index]]))[0]
        ax.scatter(
            [displayed_position[0]],
            [displayed_position[1]],
            [displayed_position[2]],
            color=color,
            edgecolors="white",
            linewidths=0.9,
            marker="X",
            s=54,
            depthshade=False,
        )

    title = f"Coverage ratio: {coverage_ratio:.2%}"
    if title_suffix:
        title = f"{title} | {title_suffix}"
    ax.set_title(title)
    handles = [
        Patch(facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_EDGE, label="obstacle"),
        Patch(facecolor=COVERED_COLOR, edgecolor=COVERED_EDGE, alpha=0.35, label="covered shell"),
        Line2D(
            [0],
            [0],
            color=FRONTIER_COLOR,
            marker="P",
            linestyle="",
            markeredgecolor=FRONTIER_EDGE,
            label="frontier",
        ),
        *[
            Line2D([0], [0], color=UAV_COLORS[index % len(UAV_COLORS)], linewidth=2.8, label=f"uav-{index}")
            for index in range(len(trajectories))
        ],
    ]
    ax.legend(handles=handles, loc="upper right")


def plot_world(world, ax=None):
    """Plot obstacle voxels for a ground-truth world."""
    if ax is None:
        figure = plt.figure(figsize=(8, 6))
        ax = figure.add_subplot(111, projection="3d")
    obstacle_mask = world.grid == int(VoxelState.OBSTACLE)
    if np.any(obstacle_mask):
        ax.voxels(
            obstacle_mask,
            facecolors=OBSTACLE_COLOR,
            edgecolor=OBSTACLE_EDGE,
            linewidth=0.35,
            alpha=0.96,
        )
    _style_axes(ax, world.grid_size)
    return ax


def plot_episode(env, save_path: str | Path | None = None):
    """Plot obstacles, covered voxels, current frontiers, and UAV trajectories."""
    if env.world is None or env.global_coverage is None:
        raise RuntimeError("Environment must be reset before rendering.")

    figure = plt.figure(figsize=(10, 7))
    ax = figure.add_subplot(111, projection="3d")
    covered_mask = env.global_coverage.base_states == int(VoxelState.COVERED)
    frontier_points = sorted({voxel for frontier_set in env.frontier_sets for voxel in frontier_set})
    frontier_mask = np.zeros(env.world.grid_size, dtype=bool)
    if frontier_points:
        frontier_indices = tuple(np.asarray(frontier_points).T)
        frontier_mask[frontier_indices] = True
    _draw_scene(
        ax,
        tuple(int(value) for value in env.world.grid_size),
        env.world.grid == int(VoxelState.OBSTACLE),
        covered_mask,
        frontier_mask,
        [np.asarray(agent.trajectory, dtype=float) for agent in env.agents],
        np.stack([agent.position for agent in env.agents]).astype(float),
        env.global_coverage.coverage_ratio(env.world),
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    return figure


def plot_episode_animation(
    replay: EpisodeReplay,
    save_path: str | Path | None = None,
    *,
    interval_ms: int = 250,
):
    """Render a full exploration replay as a GIF-friendly matplotlib animation."""
    if not replay.frames:
        raise ValueError("Episode replay must contain at least one frame.")

    figure = plt.figure(figsize=(10, 7))
    ax = figure.add_subplot(111, projection="3d")

    def _update(frame_index: int):
        ax.clear()
        frame = replay.frames[frame_index]
        trajectories = [
            np.asarray([item.uav_positions[agent_index] for item in replay.frames[: frame_index + 1]], dtype=float)
            for agent_index in range(frame.uav_positions.shape[0])
        ]
        _draw_scene(
            ax,
            replay.grid_size,
            replay.obstacle_mask,
            frame.covered_mask,
            frame.frontier_mask,
            trajectories,
            frame.uav_positions,
            frame.coverage_ratio,
            title_suffix=f"step {frame.step}",
        )
        return ()

    animation = FuncAnimation(
        figure,
        _update,
        frames=len(replay.frames),
        interval=interval_ms,
        blit=False,
        repeat_delay=800,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, round(1000 / max(interval_ms, 1)))
        animation.save(destination, writer=PillowWriter(fps=fps), dpi=120)
        plt.close(figure)
    return animation


def plot_coverage_curve(values: list[float], save_path: str | Path | None = None):
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(values)), values, color="#2563eb", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.95)
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    return figure


def plot_training_history(history: Sequence, save_path: str | Path | None = None):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    episodes = np.arange(1, len(history) + 1)
    axes[0, 0].plot(episodes, [item.reward for item in history], color="#2563eb")
    axes[0, 0].set_title("Episode reward")
    axes[0, 1].plot(episodes, [item.coverage_ratio for item in history], color="#059669")
    axes[0, 1].set_title("Coverage ratio")
    axes[1, 0].plot(episodes, [item.episode_length for item in history], color="#ea580c")
    axes[1, 0].set_title("Episode length")
    axes[1, 1].plot(episodes, [item.average_loss for item in history], color="#7c3aed")
    axes[1, 1].set_title("Average loss")
    for ax in axes.flat:
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    axes[0, 1].set_ylim(0.0, 1.0)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.94, hspace=0.3, wspace=0.25)
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    return figure


def plot_episode_diagnostics(trace, save_path: str | Path | None = None):
    steps = np.arange(len(trace.coverage_curve))
    active_agents = [sum(flags) for flags in trace.active_flags]
    figure, axes = plt.subplots(3, 2, figsize=(12, 11))
    axes[0, 0].plot(steps, trace.coverage_curve, color="#2563eb")
    axes[0, 0].set_title("Coverage over time")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 1].plot(steps, trace.team_new_coverage, color="#059669")
    axes[0, 1].set_title("New covered voxels per step")
    axes[1, 0].plot(steps, trace.frontier_counts, label="frontiers", color="#ea580c")
    axes[1, 0].plot(steps, trace.hover_counts, label="hover count", color="#dc2626")
    axes[1, 0].set_title("Exploration diagnostics")
    axes[1, 0].legend()
    axes[1, 1].plot(steps, trace.planner_failure_counts, label="planner failures", color="#dc2626")
    axes[1, 1].plot(steps, active_agents, label="active agents", color="#0891b2")
    axes[1, 1].set_title("Planner and fleet health")
    axes[1, 1].legend()
    axes[2, 0].plot(steps, trace.physical_communication_links, label="physical links", color="#7c3aed")
    axes[2, 0].plot(steps, trace.effective_communication_links, label="effective links", color="#2563eb")
    axes[2, 0].set_title("Coordination links")
    axes[2, 0].legend()
    axes[2, 1].plot(steps, trace.zero_gain_streaks, label="zero-gain streak", color="#ea580c")
    axes[2, 1].plot(steps, trace.collision_count, label="collisions", color="#111827")
    axes[2, 1].set_title("Stagnation and safety")
    axes[2, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Step")
        ax.grid(alpha=0.25)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.95, hspace=0.35, wspace=0.25)
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    return figure


def plot_evaluation_summary(history: Sequence, save_path: str | Path | None = None):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    episodes = np.arange(1, len(history) + 1)
    axes[0].bar(episodes, [item.coverage_ratio for item in history], color="#2563eb")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Coverage")
    axes[1].bar(episodes, [item.episode_length for item in history], color="#059669")
    axes[1].set_title("Episode length")
    axes[2].bar(episodes, [int(item.success) for item in history], color="#ea580c")
    axes[2].set_ylim(0.0, 1.1)
    axes[2].set_title("Success")
    for ax in axes:
        ax.set_xlabel("Episode")
        ax.grid(axis="y", alpha=0.25)
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.88, wspace=0.28)
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    return figure


def plot_planner_comparison(rows: Sequence[dict[str, float | str]], save_path: str | Path | None = None):
    planners = [str(row["planner_type"]) for row in rows]
    path_lengths = [float(row["average_path_length"]) for row in rows]
    accelerations = [float(row["mean_acceleration"]) for row in rows]
    smoothness = [float(row["smoothness_cost"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(planners, path_lengths, color="#2563eb")
    axes[0].set_title("Average path length")
    axes[1].bar(planners, accelerations, color="#059669")
    axes[1].set_title("Mean acceleration")
    axes[2].bar(planners, smoothness, color="#ea580c")
    axes[2].set_title("Smoothness cost")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.88, wspace=0.28)
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    return figure
