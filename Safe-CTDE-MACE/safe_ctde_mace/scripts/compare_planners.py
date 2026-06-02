from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np

from safe_ctde_mace.envs.multi_uav_env import MultiUAVCoverageEnv
from safe_ctde_mace.mapping.frontier_detector import CandidateFeatureLayout, candidate_score
from safe_ctde_mace.utils.config import load_config
from safe_ctde_mace.utils.replay import EpisodeReplay, capture_episode_frame, start_episode_replay
from safe_ctde_mace.utils.visualization import plot_episode, plot_episode_animation, plot_planner_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare A* and EGO-style planner diagnostics.")
    parser.add_argument(
        "--config",
        type=str,
        default="safe_ctde_mace/configs/verified_baseline.yaml",
        help="Base YAML config.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Maximum heuristic rollout steps.")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/planner_comparison", help="Output directory.")
    return parser.parse_args()


def _choose_action(observation: dict[str, np.ndarray]) -> int:
    valid = np.flatnonzero(observation["action_mask"])
    if len(valid) == 0:
        return 0
    features = observation["candidate_features"]
    layout = CandidateFeatureLayout.from_feature_width(features.shape[1])
    scores = np.asarray([candidate_score(row, layout) for row in features], dtype=float)
    scores[~observation["action_mask"]] = -np.inf
    return int(np.argmax(scores))


def _run_rollout(
    config: dict,
    planner_type: str,
    steps: int,
) -> tuple[dict[str, float | str], MultiUAVCoverageEnv, EpisodeReplay]:
    local_config = deepcopy(config)
    local_config["environment"]["planner_type"] = planner_type
    env = MultiUAVCoverageEnv(local_config)
    observations, info = env.reset(seed=int(local_config.get("seed", 0)))
    replay = start_episode_replay(env)
    path_lengths: list[float] = []
    accelerations: list[float] = []
    smoothness_costs: list[float] = []
    for _ in range(steps):
        actions = [_choose_action(observation) for observation in observations]
        observations, _, terminated, truncated, info = env.step(actions)
        path_lengths.append(float(info["average_path_length"]))
        accelerations.append(float(info["mean_acceleration"]))
        smoothness_costs.append(float(info["smoothness_cost"]))
        replay.frames.append(capture_episode_frame(env))
        if terminated or truncated:
            break
    return (
        {
            "planner_type": planner_type,
            "coverage_ratio": float(info["coverage_ratio"]),
            "average_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
            "mean_acceleration": float(np.mean(accelerations)) if accelerations else 0.0,
            "smoothness_cost": float(np.mean(smoothness_costs)) if smoothness_costs else 0.0,
            "episode_length": float(info["episode_length"]),
        },
        env,
        replay,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.artifact_dir)
    rows: list[dict[str, float | str]] = []
    for planner_type in ("astar", "ego"):
        row, env, replay = _run_rollout(config, planner_type, args.steps)
        rows.append(row)
        plot_episode(env, output_dir / f"{planner_type}_episode.png")
        plot_episode_animation(replay, output_dir / f"{planner_type}_episode.gif")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "planner_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_planner_comparison(rows, output_dir / "planner_comparison.png")
    for row in rows:
        print(
            f"planner={row['planner_type']} coverage={row['coverage_ratio']:.3f} "
            f"path={row['average_path_length']:.3f} accel={row['mean_acceleration']:.3f} "
            f"smooth={row['smoothness_cost']:.3f}"
        )


if __name__ == "__main__":
    main()
