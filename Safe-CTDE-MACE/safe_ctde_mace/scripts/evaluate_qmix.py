from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from safe_ctde_mace.marl.trainer import QMIXTrainer
from safe_ctde_mace.utils.config import load_config
from safe_ctde_mace.utils.reporting import (
    save_episode_summaries,
    save_failure_summaries,
    save_step_diagnostics,
    save_trace,
    summarize_trace_diagnostics,
)
from safe_ctde_mace.utils.visualization import (
    plot_episode,
    plot_episode_animation,
    plot_episode_diagnostics,
    plot_evaluation_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a QMIX Safe-CTDE-MACE policy.")
    parser.add_argument(
        "--config",
        type=str,
        default="safe_ctde_mace/configs/qmix_ego.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint.")
    parser.add_argument("--episodes", type=int, default=None, help="Legacy fixed-map evaluation count.")
    parser.add_argument(
        "--seed-count",
        type=int,
        default=10,
        help="Evaluate on this many consecutive seeds starting from config.seed.",
    )
    parser.add_argument("--artifact-dir", type=str, default="artifacts/qmix_evaluate", help="Report directory.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override training.device for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["training"]["device"] = args.device
    trainer = QMIXTrainer(config)
    if args.checkpoint:
        trainer.agent.load(args.checkpoint)
    evaluation_seeds = (
        list(range(int(config.get("seed", 0)), int(config.get("seed", 0)) + args.seed_count))
        if args.seed_count > 0
        else None
    )
    history = trainer.evaluate(
        episodes=args.episodes or len(evaluation_seeds or []) or 5,
        seeds=evaluation_seeds,
    )
    artifacts = Path(args.artifact_dir)
    save_episode_summaries(artifacts / "evaluation_history.csv", history)
    plot_evaluation_summary(history, artifacts / "evaluation_summary.png")
    if trainer.evaluation_traces:
        save_trace(artifacts / "last_evaluation_trace.json", trainer.evaluation_traces[-1])
        save_step_diagnostics(artifacts / "last_evaluation_step_diagnostics.csv", trainer.evaluation_traces[-1])
        save_failure_summaries(
            artifacts / "evaluation_failure_summary.csv",
            history,
            trainer.evaluation_traces,
        )
        plot_episode_diagnostics(trainer.evaluation_traces[-1], artifacts / "last_evaluation_diagnostics.png")
        plot_episode(trainer.env, artifacts / "last_evaluation_episode.png")
    if trainer.evaluation_replays:
        plot_episode_animation(trainer.evaluation_replays[-1], artifacts / "last_evaluation_replay.gif")
    print(
        f"episodes={len(history)} "
        f"coverage_mean={mean(item.coverage_ratio for item in history):.3f} "
        f"success_rate={mean(float(item.success) for item in history):.3f} "
        f"episode_length_mean={mean(item.episode_length for item in history):.1f} "
        f"artifacts={artifacts}"
    )
    if trainer.evaluation_traces:
        diagnostics = summarize_trace_diagnostics(trainer.evaluation_traces[-1])
        print(
            "last_trace "
            f"plateau_step={diagnostics['plateau_step']} "
            f"first_hover_step={diagnostics['first_hover_step']} "
            f"first_collision_step={diagnostics['first_collision_step']} "
            f"first_planner_failure_step={diagnostics['first_planner_failure_step']} "
            f"first_physical_link_step={diagnostics['first_physical_link_step']} "
            f"first_effective_link_step={diagnostics['first_effective_link_step']} "
            f"max_zero_gain_streak={diagnostics['max_zero_gain_streak']} "
            f"planner_failures={diagnostics['planner_failure_total']} "
            f"physical_links_mean={diagnostics['physical_links_mean']:.2f} "
            f"effective_links_mean={diagnostics['effective_links_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
