from __future__ import annotations

import numpy as np
import torch
from torch import nn


def flatten_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten one structured UAV observation into a 1D float vector."""
    parts = [
        observation["local_voxel_map"].reshape(-1),
        observation["self_state"].reshape(-1),
        observation["neighbor_states"].reshape(-1),
        observation["coverage_ratio"].reshape(-1),
        observation["candidate_features"].reshape(-1),
    ]
    return np.concatenate(parts).astype(np.float32)


def observation_dim(observation: dict[str, np.ndarray]) -> int:
    return int(flatten_observation(observation).shape[0])


class QNetwork(nn.Module):
    """Lightweight MLP baseline for shared DQN."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class QMixer(nn.Module):
    """Monotonic QMIX mixer conditioned on a centralized global state."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        hidden_dim: int = 64,
        hypernet_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, self.num_agents * hidden_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size = agent_qs.shape[0]
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)
        weights_1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.num_agents, self.hidden_dim)
        bias_1 = self.hyper_b1(states).view(batch_size, 1, self.hidden_dim)
        hidden = torch.nn.functional.elu(torch.bmm(agent_qs, weights_1) + bias_1)
        weights_2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.hidden_dim, 1)
        bias_2 = self.hyper_b2(states).view(batch_size, 1, 1)
        total_q = torch.bmm(hidden, weights_2) + bias_2
        return total_q.view(batch_size)
