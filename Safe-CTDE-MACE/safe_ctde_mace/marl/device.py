from __future__ import annotations

from typing import Literal

import torch


DevicePreference = Literal["auto", "cpu", "cuda"]


def resolve_device(preference: str | None = None) -> torch.device:
    """Resolve a user-facing device preference into a concrete torch device."""
    normalized = str(preference or "auto").lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be one of: auto, cpu, cuda.")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the current PyTorch runtime.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
