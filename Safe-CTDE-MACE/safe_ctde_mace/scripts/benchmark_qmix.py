from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from safe_ctde_mace.marl.trainer import QMIXTrainer
from safe_ctde_mace.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short QMIX throughput benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default="safe_ctde_mace/configs/qmix_ego_large.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument("--episodes", type=int, default=2, help="Number of training episodes to time.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override training.device for this run.",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="Override training.num_envs for this run.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON result path.")
    parser.add_argument("--append-csv", type=str, default=None, help="Optional CSV path to append one row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = deepcopy(load_config(args.config))
    config["training"]["save_interval"] = 0
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.num_envs is not None:
        config["training"]["num_envs"] = args.num_envs

    trainer = QMIXTrainer(config)
    start = perf_counter()
    history = trainer.train(args.episodes)
    wall_time = perf_counter() - start
    env_steps = sum(item.episode_length for item in history)
    result = {
        "device": str(trainer.agent.device),
        "episodes": len(history),
        "num_envs": int(config["training"].get("num_envs", 1)),
        "env_steps": int(env_steps),
        "wall_time": float(wall_time),
        "steps_per_second": float(env_steps / wall_time) if wall_time > 0 else 0.0,
        "episodes_per_hour": float(len(history) * 3600.0 / wall_time) if wall_time > 0 else 0.0,
    }
    print(json.dumps(result, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.append_csv:
        csv_path = Path(args.append_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            if write_header:
                writer.writeheader()
            writer.writerow(result)


if __name__ == "__main__":
    main()
