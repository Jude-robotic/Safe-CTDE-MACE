from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from safe_ctde_mace.envs.multi_uav_env import MultiUAVCoverageEnv
from safe_ctde_mace.mapping.frontier_detector import CandidateFeatureLayout, candidate_score
from safe_ctde_mace.utils.config import load_config
from safe_ctde_mace.utils.visualization import plot_coverage_curve, plot_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a heuristic frontier policy demo.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--steps", type=int, default=30, help="Maximum demo steps.")
    parser.add_argument("--save-dir", type=str, default="artifacts/heuristic_demo", help="Artifact directory.")
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


def main() -> None:
    args = parse_args()
    env = MultiUAVCoverageEnv(load_config(args.config))
    observations, info = env.reset()
    coverage_curve = [info["coverage_ratio"]]

    for _ in range(args.steps):
        actions = [_choose_action(observation) for observation in observations]
        observations, _, terminated, truncated, info = env.step(actions)
        coverage_curve.append(info["coverage_ratio"])
        if terminated or truncated:
            break

    output_dir = Path(args.save_dir)
    plot_episode(env, output_dir / "episode.png")
    plot_coverage_curve(coverage_curve, output_dir / "coverage_curve.png")
    print(
        f"steps={info['episode_length']} coverage={info['coverage_ratio']:.3f} "
        f"success={info['success']}"
    )


if __name__ == "__main__":
    main()
