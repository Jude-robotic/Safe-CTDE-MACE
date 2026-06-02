from __future__ import annotations

import argparse

import torch

from safe_ctde_mace.marl.device import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the active PyTorch runtime and resolved device.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device preference to resolve.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_device(args.device)
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_version={torch.version.cuda}")
    print(f"device_count={torch.cuda.device_count()}")
    print(f"resolved_device={resolved}")
    if torch.cuda.is_available():
        print(f"device_name={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
