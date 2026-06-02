from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from safe_ctde_mace.marl.trainer import QMIXTrainer
from safe_ctde_mace.utils.config import load_config
from safe_ctde_mace.utils.reporting import save_episode_summaries, save_trace
from safe_ctde_mace.utils.visualization import (
    plot_episode,
    plot_episode_animation,
    plot_episode_diagnostics,
    plot_evaluation_summary,
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the QMIX Safe-CTDE-MACE branch.")
    parser.add_argument(
        "--config",
        type=str,
        default="safe_ctde_mace/configs/qmix_ego.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes.")
    parser.add_argument("--output", type=str, default="checkpoints/qmix_final.pt", help="Final model path.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Legacy fixed-map evaluation count.")
    parser.add_argument(
        "--eval-seed-count",
        type=int,
        default=10,
        help="Evaluate on this many consecutive seeds starting from config.seed.",
    )
    parser.add_argument("--artifact-dir", type=str, default="artifacts/qmix_train", help="Directory for reports.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override training.device for this run.",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="Override training.num_envs for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.num_envs is not None:
        config["training"]["num_envs"] = args.num_envs
    trainer = QMIXTrainer(config)
    history = trainer.train(args.episodes)
    trainer.agent.save(Path(args.output))
    artifacts = Path(args.artifact_dir)
    save_episode_summaries(artifacts / "train_history.csv", history)
    plot_training_history(history, artifacts / "training_curves.png")
    if trainer.last_trace is not None:
        save_trace(artifacts / "last_train_trace.json", trainer.last_trace)
        plot_episode_diagnostics(trainer.last_trace, artifacts / "last_train_diagnostics.png")

    evaluation_seeds = (
        list(range(int(config.get("seed", 0)), int(config.get("seed", 0)) + args.eval_seed_count))
        if args.eval_seed_count > 0
        else None
    )
    evaluation = trainer.evaluate(
        episodes=args.eval_episodes or len(evaluation_seeds or []) or 3,
        seeds=evaluation_seeds,
    )
    save_episode_summaries(artifacts / "evaluation_history.csv", evaluation)
    plot_evaluation_summary(evaluation, artifacts / "evaluation_summary.png")
    if trainer.evaluation_traces:
        save_trace(artifacts / "evaluation_trace.json", trainer.evaluation_traces[-1])
        plot_episode_diagnostics(trainer.evaluation_traces[-1], artifacts / "evaluation_diagnostics.png")
        plot_episode(trainer.env, artifacts / "evaluation_episode.png")
    if trainer.evaluation_replays:
        plot_episode_animation(trainer.evaluation_replays[-1], artifacts / "evaluation_replay.gif")

    last = history[-1]
    eval_success = sum(item.success for item in evaluation) / max(len(evaluation), 1)
    eval_coverage = sum(item.coverage_ratio for item in evaluation) / max(len(evaluation), 1)
    print(
        f"episodes={len(history)} reward={last.reward:.3f} "
        f"coverage={last.coverage_ratio:.3f} length={last.episode_length} success={last.success} "
        f"eval_coverage={eval_coverage:.3f} eval_success_rate={eval_success:.3f} "
        f"artifacts={artifacts}"
    )


if __name__ == "__main__":
    main()
