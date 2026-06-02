from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from safe_ctde_mace.marl.trainer import Trainer
from safe_ctde_mace.utils.config import load_config
from safe_ctde_mace.utils.reporting import save_episode_summaries, save_trace
from safe_ctde_mace.utils.visualization import (
    plot_episode,
    plot_episode_animation,
    plot_episode_diagnostics,
    plot_evaluation_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a shared DQN Safe-CTDE-MACE policy.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/evaluate", help="Directory for reports and plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = Trainer(load_config(args.config))
    if args.checkpoint:
        trainer.agent.load(args.checkpoint)
    history = trainer.evaluate(args.episodes)
    artifacts = Path(args.artifact_dir)
    save_episode_summaries(artifacts / "evaluation_history.csv", history)
    plot_evaluation_summary(history, artifacts / "evaluation_summary.png")
    if trainer.evaluation_traces:
        save_trace(artifacts / "last_evaluation_trace.json", trainer.evaluation_traces[-1])
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


if __name__ == "__main__":
    main()
