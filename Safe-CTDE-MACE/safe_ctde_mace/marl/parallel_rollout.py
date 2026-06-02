from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from typing import Any
import multiprocessing as mp

import numpy as np

from safe_ctde_mace.envs.multi_uav_env import MultiUAVCoverageEnv


@dataclass(slots=True)
class WorkerReady:
    observations: list[dict[str, np.ndarray]]
    state: np.ndarray
    reset_info: dict[str, Any]


@dataclass(slots=True)
class WorkerStep:
    next_observations: list[dict[str, np.ndarray]]
    next_state: np.ndarray
    rewards: list[float]
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    reset_observations: list[dict[str, np.ndarray]] | None
    reset_state: np.ndarray | None
    reset_info: dict[str, Any] | None


def _worker_loop(config: dict[str, Any], worker_id: int, seed_stride: int, connection: Connection) -> None:
    env = MultiUAVCoverageEnv(config)
    local_episode_index = 0
    base_seed = int(config.get("seed", 0))

    observations, reset_info = env.reset(seed=base_seed + worker_id * seed_stride + local_episode_index)
    connection.send(WorkerReady(observations, env.get_global_state(), reset_info))
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command != "step":
                raise ValueError(f"Unknown worker command: {command}")

            next_observations, rewards, terminated, truncated, info = env.step(payload)
            next_state = env.get_global_state()
            reset_observations = None
            reset_state = None
            next_reset_info = None
            if terminated or truncated:
                local_episode_index += 1
                reset_observations, next_reset_info = env.reset(
                    seed=base_seed + worker_id * seed_stride + local_episode_index
                )
                reset_state = env.get_global_state()
            connection.send(
                WorkerStep(
                    next_observations=next_observations,
                    next_state=next_state,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    reset_observations=reset_observations,
                    reset_state=reset_state,
                    reset_info=next_reset_info,
                )
            )
    finally:
        connection.close()


class ParallelRolloutManager:
    """Synchronous multi-process environment manager for QMIX rollouts."""

    def __init__(self, config: dict[str, Any], num_envs: int, seed_stride: int = 100_000) -> None:
        self.config = config
        self.num_envs = int(num_envs)
        self.seed_stride = int(seed_stride)
        self.context: BaseContext = mp.get_context("spawn")
        self.connections: list[Connection] = []
        self.processes: list[mp.Process] = []

    def start(self) -> list[WorkerReady]:
        ready: list[WorkerReady] = []
        for worker_id in range(self.num_envs):
            parent_conn, child_conn = self.context.Pipe()
            process = self.context.Process(
                target=_worker_loop,
                args=(self.config, worker_id, self.seed_stride, child_conn),
            )
            process.start()
            child_conn.close()
            self.connections.append(parent_conn)
            self.processes.append(process)
        for connection in self.connections:
            ready.append(connection.recv())
        return ready

    def step(self, action_batches: list[list[int]]) -> list[WorkerStep]:
        for connection, actions in zip(self.connections, action_batches, strict=True):
            connection.send(("step", actions))
        return [connection.recv() for connection in self.connections]

    def close(self) -> None:
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    def __enter__(self) -> "ParallelRolloutManager":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
