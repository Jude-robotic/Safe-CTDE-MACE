from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from safe_ctde_mace.marl.networks import QNetwork, flatten_observation


class SharedDQNAgent:
    """Parameter-sharing DQN policy shared by all UAVs."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        training_config: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.config = training_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q_network = QNetwork(
            obs_dim,
            num_actions,
            hidden_dim=int(training_config.get("hidden_dim", 256)),
        ).to(self.device)
        self.target_network = QNetwork(
            obs_dim,
            num_actions,
            hidden_dim=int(training_config.get("hidden_dim", 256)),
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=float(training_config["learning_rate"]),
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = float(training_config["gamma"])
        self.target_update_interval = int(training_config["target_update_interval"])
        self.steps_done = 0
        self.training_steps = 0

    @property
    def epsilon(self) -> float:
        start = float(self.config["epsilon_start"])
        end = float(self.config["epsilon_end"])
        decay_steps = max(int(self.config["epsilon_decay_steps"]), 1)
        fraction = min(self.steps_done / decay_steps, 1.0)
        return start + fraction * (end - start)

    def select_action(
        self,
        observation: dict[str, np.ndarray],
        explore: bool = True,
    ) -> int:
        mask = observation["action_mask"].astype(bool)
        valid_actions = np.flatnonzero(mask)
        if len(valid_actions) == 0:
            return 0

        epsilon = self.epsilon if explore else 0.0
        if explore and np.random.random() < epsilon:
            action = int(np.random.choice(valid_actions))
        else:
            obs_tensor = torch.from_numpy(flatten_observation(observation)).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor).squeeze(0).cpu().numpy()
            q_values[~mask] = -np.inf
            action = int(np.argmax(q_values))
        if explore:
            self.steps_done += 1
        return action

    def select_actions(
        self,
        observations: list[dict[str, np.ndarray]],
        explore: bool = True,
    ) -> list[int]:
        return [self.select_action(observation, explore=explore) for observation in observations]

    def train_step(self, batch: dict[str, np.ndarray]) -> float:
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)
        next_masks = torch.from_numpy(batch["next_action_masks"]).to(self.device)

        q_values = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            online_next_q = self.q_network(next_obs)
            online_next_q = online_next_q.masked_fill(~next_masks, float("-inf"))
            valid_next = next_masks.any(dim=1)
            next_actions = torch.argmax(online_next_q, dim=1)
            target_next_q = self.target_network(next_obs).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
            target_next_q = torch.where(valid_next, target_next_q, torch.zeros_like(target_next_q))
            targets = rewards + self.gamma * (1.0 - dones) * target_next_q

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.training_steps += 1

        if self.training_steps % self.target_update_interval == 0:
            self.update_target_network()
        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "training_steps": self.training_steps,
                "config": self.config,
            },
            destination,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        self.q_network.load_state_dict(payload["q_network"])
        self.target_network.load_state_dict(payload["target_network"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.steps_done = int(payload["steps_done"])
        self.training_steps = int(payload["training_steps"])
