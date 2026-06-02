from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from safe_ctde_mace.mapping.frontier_detector import CANDIDATE_FEATURE_SCHEMA_VERSION
from safe_ctde_mace.marl.device import resolve_device
from safe_ctde_mace.marl.networks import QMixer, QNetwork, flatten_observation


class QMIXAgent:
    """Parameter-sharing QMIX agent for centralized training and decentralized execution."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_agents: int,
        num_actions: int,
        training_config: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.num_agents = int(num_agents)
        self.num_actions = int(num_actions)
        self.config = training_config
        configured_device = device or training_config.get("device", "auto")
        self.device = resolve_device(str(configured_device))
        hidden_dim = int(training_config.get("hidden_dim", 256))
        mixer_hidden_dim = int(training_config.get("mixer_hidden_dim", 64))
        hypernet_hidden_dim = int(training_config.get("hypernet_hidden_dim", 128))

        self.q_network = QNetwork(self.obs_dim, self.num_actions, hidden_dim=hidden_dim).to(self.device)
        self.target_q_network = QNetwork(self.obs_dim, self.num_actions, hidden_dim=hidden_dim).to(self.device)
        self.mixer = QMixer(
            self.num_agents,
            self.state_dim,
            hidden_dim=mixer_hidden_dim,
            hypernet_hidden_dim=hypernet_hidden_dim,
        ).to(self.device)
        self.target_mixer = QMixer(
            self.num_agents,
            self.state_dim,
            hidden_dim=mixer_hidden_dim,
            hypernet_hidden_dim=hypernet_hidden_dim,
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
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

    def select_actions(
        self,
        observations: list[dict[str, np.ndarray]],
        explore: bool = True,
    ) -> list[int]:
        return self.select_actions_batch([observations], explore=explore)[0]

    def select_actions_batch(
        self,
        observation_batches: list[list[dict[str, np.ndarray]]],
        explore: bool = True,
    ) -> list[list[int]]:
        actions = [[0 for _ in observations] for observations in observation_batches]
        epsilons = [
            self._epsilon_at_step(self.steps_done + batch_index) if explore else 0.0
            for batch_index in range(len(observation_batches))
        ]
        greedy_indices: list[int] = []
        flattened: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        flat_locations: list[tuple[int, int]] = []
        for batch_index, observations in enumerate(observation_batches):
            for agent_index, observation in enumerate(observations):
                mask = observation["action_mask"].astype(bool)
                valid_actions = np.flatnonzero(mask)
                if len(valid_actions) == 0:
                    continue
                if explore and np.random.random() < epsilons[batch_index]:
                    actions[batch_index][agent_index] = int(np.random.choice(valid_actions))
                else:
                    greedy_indices.append(len(flat_locations))
                    flattened.append(flatten_observation(observation))
                    masks.append(mask)
                    flat_locations.append((batch_index, agent_index))

        if greedy_indices:
            obs_tensor = self._tensor(np.stack(flattened))
            with torch.no_grad():
                q_batch = self.q_network(obs_tensor).cpu().numpy()
            for flat_index, q_values, mask in zip(greedy_indices, q_batch, masks, strict=True):
                batch_index, agent_index = flat_locations[flat_index]
                q_values[~mask] = -np.inf
                actions[batch_index][agent_index] = int(np.argmax(q_values))
        if explore:
            self.steps_done += len(observation_batches)
        return actions

    def _epsilon_at_step(self, step: int) -> float:
        start = float(self.config["epsilon_start"])
        end = float(self.config["epsilon_end"])
        decay_steps = max(int(self.config["epsilon_decay_steps"]), 1)
        fraction = min(step / decay_steps, 1.0)
        return start + fraction * (end - start)

    def train_step(self, batch: dict[str, np.ndarray]) -> float:
        obs = self._tensor(batch["obs"])
        actions = self._tensor(batch["actions"], dtype=torch.long)
        states = self._tensor(batch["states"])
        rewards = self._tensor(batch["rewards"])
        next_obs = self._tensor(batch["next_obs"])
        next_states = self._tensor(batch["next_states"])
        dones = self._tensor(batch["dones"])
        next_masks = self._tensor(batch["next_action_masks"])

        batch_size = obs.shape[0]
        q_values = self.q_network(obs.view(batch_size * self.num_agents, -1)).view(
            batch_size,
            self.num_agents,
            self.num_actions,
        )
        chosen_qs = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        total_q = self.mixer(chosen_qs, states)

        with torch.no_grad():
            online_next_q = self.q_network(next_obs.view(batch_size * self.num_agents, -1)).view(
                batch_size,
                self.num_agents,
                self.num_actions,
            )
            online_next_q = online_next_q.masked_fill(~next_masks, float("-inf"))
            valid_next = next_masks.any(dim=2)
            next_actions = torch.argmax(online_next_q, dim=2)
            target_next_q = self.target_q_network(
                next_obs.view(batch_size * self.num_agents, -1)
            ).view(batch_size, self.num_agents, self.num_actions)
            target_agent_qs = target_next_q.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
            target_agent_qs = torch.where(valid_next, target_agent_qs, torch.zeros_like(target_agent_qs))
            target_total_q = self.target_mixer(target_agent_qs, next_states)
            targets = rewards + self.gamma * (1.0 - dones) * target_total_q

        loss = self.loss_fn(total_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()
        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.update_target_network()
        return float(loss.item())

    def _tensor(self, array: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=self.device)

    def update_target_network(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": self._architecture_metadata(),
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "training_steps": self.training_steps,
                "config": self.config,
            },
            destination,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        self._validate_checkpoint_metadata(payload)
        self.q_network.load_state_dict(payload["q_network"])
        self.target_q_network.load_state_dict(payload["target_q_network"])
        self.mixer.load_state_dict(payload["mixer"])
        self.target_mixer.load_state_dict(payload["target_mixer"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.steps_done = int(payload["steps_done"])
        self.training_steps = int(payload["training_steps"])

    def _architecture_metadata(self) -> dict[str, int]:
        return {
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "num_agents": self.num_agents,
            "num_actions": self.num_actions,
            "feature_schema_version": CANDIDATE_FEATURE_SCHEMA_VERSION,
        }

    def _validate_checkpoint_metadata(self, payload: dict[str, Any]) -> None:
        metadata = dict(payload.get("metadata") or self._infer_checkpoint_metadata(payload))
        metadata.setdefault("feature_schema_version", 1)
        expected = self._architecture_metadata()
        mismatches = [
            f"{name}: checkpoint={int(metadata[name])}, current={expected[name]}"
            for name in expected
            if int(metadata[name]) != expected[name]
        ]
        if mismatches:
            mismatch_text = "; ".join(mismatches)
            raise ValueError(
                "QMIX checkpoint architecture mismatch. "
                f"Use the checkpoint produced by the same scenario config. {mismatch_text}"
            )

    @staticmethod
    def _infer_checkpoint_metadata(payload: dict[str, Any]) -> dict[str, int]:
        q_network = payload["q_network"]
        mixer = payload["mixer"]
        hidden_dim = int(mixer["hyper_b1.weight"].shape[0])
        return {
            "obs_dim": int(q_network["model.0.weight"].shape[1]),
            "state_dim": int(mixer["hyper_w1.0.weight"].shape[1]),
            "num_agents": int(mixer["hyper_w1.2.weight"].shape[0] // hidden_dim),
            "num_actions": int(q_network["model.4.weight"].shape[0]),
            "feature_schema_version": 1,
        }
