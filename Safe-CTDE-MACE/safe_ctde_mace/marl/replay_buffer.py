from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np


class ReplayBuffer:
    """Simple replay buffer storing per-agent transitions."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._buffer: Deque[tuple[np.ndarray, int, float, np.ndarray, float, np.ndarray, np.ndarray]] = deque(
            maxlen=self.capacity
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self._buffer.append(
            (
                obs.astype(np.float32),
                int(action),
                float(reward),
                next_obs.astype(np.float32),
                float(done),
                action_mask.astype(bool),
                next_action_mask.astype(bool),
            )
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size > len(self._buffer):
            raise ValueError("Cannot sample more items than are stored in the buffer.")
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[index] for index in indices]
        obs, actions, rewards, next_obs, dones, masks, next_masks = zip(*batch, strict=True)
        return {
            "obs": np.stack(obs),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_obs": np.stack(next_obs),
            "dones": np.asarray(dones, dtype=np.float32),
            "action_masks": np.stack(masks),
            "next_action_masks": np.stack(next_masks),
        }


class JointReplayBuffer:
    """Replay buffer storing one joint multi-agent transition per environment step."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._buffer: Deque[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
                np.ndarray,
                np.ndarray,
            ]
        ] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        state: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        dones: np.ndarray,
        done: bool,
        action_masks: np.ndarray,
        next_action_masks: np.ndarray,
    ) -> None:
        self._buffer.append(
            (
                obs.astype(np.float32),
                actions.astype(np.int64),
                state.astype(np.float32),
                float(reward),
                next_obs.astype(np.float32),
                next_state.astype(np.float32),
                dones.astype(np.float32),
                float(done),
                action_masks.astype(bool),
                next_action_masks.astype(bool),
            )
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size > len(self._buffer):
            raise ValueError("Cannot sample more items than are stored in the buffer.")
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[index] for index in indices]
        obs, actions, states, rewards, next_obs, next_states, dones, done_flags, masks, next_masks = zip(
            *batch,
            strict=True,
        )
        return {
            "obs": np.stack(obs),
            "actions": np.stack(actions),
            "states": np.stack(states),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_obs": np.stack(next_obs),
            "next_states": np.stack(next_states),
            "agent_dones": np.stack(dones),
            "dones": np.asarray(done_flags, dtype=np.float32),
            "action_masks": np.stack(masks),
            "next_action_masks": np.stack(next_masks),
        }
