from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from safe_ctde_mace.envs.multi_uav_env import MultiUAVCoverageEnv
from safe_ctde_mace.marl.networks import flatten_observation, observation_dim
from safe_ctde_mace.marl.parallel_rollout import ParallelRolloutManager
from safe_ctde_mace.marl.qmix import QMIXAgent
from safe_ctde_mace.marl.replay_buffer import JointReplayBuffer, ReplayBuffer
from safe_ctde_mace.marl.shared_dqn import SharedDQNAgent
from safe_ctde_mace.utils.replay import EpisodeReplay, capture_episode_frame, start_episode_replay


@dataclass(slots=True)
class EpisodeSummary:
    reward: float
    coverage_ratio: float
    episode_length: int
    success: bool
    average_loss: float
    collision_count: int
    repeated_coverage_ratio: float
    truncated: bool
    termination_reason: str
    mean_acceleration: float = 0.0
    max_acceleration: float = 0.0
    smoothness_cost: float = 0.0


@dataclass(slots=True)
class EpisodeTrace:
    coverage_curve: list[float]
    team_new_coverage: list[int]
    repeated_coverage_ratio: list[float]
    communication_links: list[int]
    physical_communication_links: list[int]
    effective_communication_links: list[int]
    global_sync_applied: list[bool]
    collision_count: list[int]
    frontier_counts: list[int]
    hover_counts: list[int]
    adjusted_counts: list[int]
    planner_failure_counts: list[int]
    zero_gain_streaks: list[int]
    planner_statuses: list[list[str]]
    shield_statuses: list[list[str]]
    active_flags: list[list[bool]]
    hover_reasons: list[list[str]] = field(default_factory=list)
    goal_conflict_resolutions: list[int] = field(default_factory=list)
    late_reassignment_applied: list[bool] = field(default_factory=list)


class Trainer:
    """Training loop for the shared DQN baseline."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.training_cfg = config["training"]
        self.env = MultiUAVCoverageEnv(config)
        observations, _ = self.env.reset()
        self.obs_dim = observation_dim(observations[0])
        self.num_actions = int(config["environment"]["num_frontier_candidates"])
        self.agent = SharedDQNAgent(self.obs_dim, self.num_actions, self.training_cfg)
        self.buffer = ReplayBuffer(self.training_cfg["replay_capacity"])
        self.last_trace: EpisodeTrace | None = None
        self.last_replay: EpisodeReplay | None = None
        self.best_trace: EpisodeTrace | None = None
        self.evaluation_traces: list[EpisodeTrace] = []
        self.evaluation_replays: list[EpisodeReplay] = []

    def run_episode(self, training: bool = True, capture_replay: bool = False) -> EpisodeSummary:
        observations, reset_info = self.env.reset()
        total_reward = 0.0
        losses: list[float] = []
        terminated = False
        truncated = False
        info = {}
        trace = EpisodeTrace(
            coverage_curve=[float(reset_info["coverage_ratio"])],
            team_new_coverage=[0],
            repeated_coverage_ratio=[0.0],
            communication_links=[int(reset_info["communication_links"])],
            physical_communication_links=[int(reset_info["physical_communication_links"])],
            effective_communication_links=[int(reset_info["effective_communication_links"])],
            global_sync_applied=[bool(reset_info["global_sync_applied"])],
            collision_count=[0],
            frontier_counts=[int(sum(reset_info["frontier_counts"]))],
            hover_counts=[0],
            hover_reasons=[list(reset_info["hover_reasons"])],
            goal_conflict_resolutions=[int(reset_info["goal_conflict_resolutions"])],
            late_reassignment_applied=[bool(reset_info["late_reassignment_applied"])],
            adjusted_counts=[0],
            planner_failure_counts=[int(reset_info["planner_failure_count"])],
            zero_gain_streaks=[int(reset_info["zero_gain_streak"])],
            planner_statuses=[list(reset_info["planner_statuses"])],
            shield_statuses=[list(reset_info["shield_statuses"])],
            active_flags=[list(reset_info["active_flags"])],
        )
        replay = start_episode_replay(self.env) if capture_replay else None

        while not (terminated or truncated):
            actions = self.agent.select_actions(observations, explore=training)
            next_observations, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated

            if training:
                for obs, action, reward, next_obs in zip(
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    strict=True,
                ):
                    self.buffer.add(
                        flatten_observation(obs),
                        action,
                        reward,
                        flatten_observation(next_obs),
                        done,
                        obs["action_mask"],
                        next_obs["action_mask"],
                    )
                if len(self.buffer) >= max(
                    int(self.training_cfg["warmup_steps"]),
                    int(self.training_cfg["batch_size"]),
                ):
                    batch = self.buffer.sample(int(self.training_cfg["batch_size"]))
                    losses.append(self.agent.train_step(batch))

            total_reward += float(np.sum(rewards))
            statuses = info.get("shield_statuses", [])
            trace.coverage_curve.append(float(info.get("coverage_ratio", 0.0)))
            trace.team_new_coverage.append(int(info.get("team_new_coverage", 0)))
            trace.repeated_coverage_ratio.append(float(info.get("repeated_coverage_ratio", 0.0)))
            trace.communication_links.append(int(info.get("communication_links", 0)))
            trace.physical_communication_links.append(int(info.get("physical_communication_links", 0)))
            trace.effective_communication_links.append(int(info.get("effective_communication_links", 0)))
            trace.global_sync_applied.append(bool(info.get("global_sync_applied", False)))
            trace.collision_count.append(int(info.get("collision_count", 0)))
            trace.frontier_counts.append(int(sum(info.get("frontier_counts", []))))
            trace.hover_counts.append(sum(status == "hover" for status in statuses))
            trace.hover_reasons.append(list(info.get("hover_reasons", [])))
            trace.goal_conflict_resolutions.append(int(info.get("goal_conflict_resolutions", 0)))
            trace.late_reassignment_applied.append(bool(info.get("late_reassignment_applied", False)))
            trace.adjusted_counts.append(sum(status == "adjusted" for status in statuses))
            trace.planner_failure_counts.append(int(info.get("planner_failure_count", 0)))
            trace.zero_gain_streaks.append(int(info.get("zero_gain_streak", 0)))
            trace.planner_statuses.append(list(info.get("planner_statuses", [])))
            trace.shield_statuses.append(list(statuses))
            trace.active_flags.append(list(info.get("active_flags", [])))
            if replay is not None:
                replay.frames.append(capture_episode_frame(self.env))
            observations = next_observations

        self.last_trace = trace
        self.last_replay = replay
        return EpisodeSummary(
            reward=total_reward,
            coverage_ratio=float(info.get("coverage_ratio", 0.0)),
            episode_length=int(info.get("episode_length", 0)),
            success=bool(info.get("success", False)),
            average_loss=float(np.mean(losses)) if losses else 0.0,
            collision_count=int(info.get("collision_count", 0)),
            repeated_coverage_ratio=float(info.get("repeated_coverage_ratio", 0.0)),
            truncated=bool(info.get("truncated", False)),
            termination_reason=str(info.get("termination_reason", "unknown")),
            mean_acceleration=float(info.get("mean_acceleration", 0.0)),
            max_acceleration=float(info.get("max_acceleration", 0.0)),
            smoothness_cost=float(info.get("smoothness_cost", 0.0)),
        )

    def train(self, num_episodes: int | None = None) -> list[EpisodeSummary]:
        episodes = int(num_episodes or self.training_cfg["num_episodes"])
        history: list[EpisodeSummary] = []
        checkpoint_dir = Path(self.training_cfg["checkpoint_dir"])
        save_interval = int(self.training_cfg["save_interval"])
        for episode in range(1, episodes + 1):
            summary = self.run_episode(training=True)
            history.append(summary)
            if self.best_trace is None or summary.coverage_ratio >= max(item.coverage_ratio for item in history[:-1]):
                self.best_trace = self.last_trace
            if save_interval > 0 and episode % save_interval == 0:
                self.agent.save(checkpoint_dir / f"shared_dqn_ep{episode}.pt")
        return history

    def evaluate(self, episodes: int = 5, capture_replay: bool = True) -> list[EpisodeSummary]:
        history: list[EpisodeSummary] = []
        self.evaluation_traces = []
        self.evaluation_replays = []
        for _ in range(episodes):
            history.append(self.run_episode(training=False, capture_replay=capture_replay))
            if self.last_trace is not None:
                self.evaluation_traces.append(self.last_trace)
            if self.last_replay is not None:
                self.evaluation_replays.append(self.last_replay)
        return history


class QMIXTrainer:
    """Training loop for the centralized-training QMIX branch."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.training_cfg = config["training"]
        self.base_seed = int(config.get("seed", 0))
        self.env = MultiUAVCoverageEnv(config)
        observations, _ = self.env.reset()
        self.obs_dim = observation_dim(observations[0])
        self.state_dim = int(self.env.get_global_state().shape[0])
        self.num_agents = int(config["environment"]["num_uavs"])
        self.num_actions = int(config["environment"]["num_frontier_candidates"])
        self.num_envs = int(self.training_cfg.get("num_envs", 1))
        self.agent = QMIXAgent(
            self.obs_dim,
            self.state_dim,
            self.num_agents,
            self.num_actions,
            self.training_cfg,
        )
        self.buffer = JointReplayBuffer(self.training_cfg["replay_capacity"])
        self.last_trace: EpisodeTrace | None = None
        self.last_replay: EpisodeReplay | None = None
        self.evaluation_traces: list[EpisodeTrace] = []
        self.evaluation_replays: list[EpisodeReplay] = []

    def run_episode(
        self,
        training: bool = True,
        capture_replay: bool = False,
        seed: int | None = None,
    ) -> EpisodeSummary:
        observations, reset_info = self.env.reset(seed=seed)
        total_reward = 0.0
        losses: list[float] = []
        terminated = False
        truncated = False
        info = {}
        trace = EpisodeTrace(
            coverage_curve=[float(reset_info["coverage_ratio"])],
            team_new_coverage=[0],
            repeated_coverage_ratio=[0.0],
            communication_links=[int(reset_info["communication_links"])],
            physical_communication_links=[int(reset_info["physical_communication_links"])],
            effective_communication_links=[int(reset_info["effective_communication_links"])],
            global_sync_applied=[bool(reset_info["global_sync_applied"])],
            collision_count=[0],
            frontier_counts=[int(sum(reset_info["frontier_counts"]))],
            hover_counts=[0],
            hover_reasons=[list(reset_info["hover_reasons"])],
            goal_conflict_resolutions=[int(reset_info["goal_conflict_resolutions"])],
            late_reassignment_applied=[bool(reset_info["late_reassignment_applied"])],
            adjusted_counts=[0],
            planner_failure_counts=[int(reset_info["planner_failure_count"])],
            zero_gain_streaks=[int(reset_info["zero_gain_streak"])],
            planner_statuses=[list(reset_info["planner_statuses"])],
            shield_statuses=[list(reset_info["shield_statuses"])],
            active_flags=[list(reset_info["active_flags"])],
        )
        replay = start_episode_replay(self.env) if capture_replay else None

        while not (terminated or truncated):
            state = self.env.get_global_state()
            actions = self.agent.select_actions(observations, explore=training)
            next_observations, rewards, terminated, truncated, info = self.env.step(actions)
            next_state = self.env.get_global_state()
            done = terminated or truncated
            team_reward = float(np.sum(rewards))

            if training:
                self.buffer.add(
                    np.stack([flatten_observation(obs) for obs in observations]),
                    np.asarray(actions, dtype=np.int64),
                    state,
                    team_reward,
                    np.stack([flatten_observation(obs) for obs in next_observations]),
                    next_state,
                    np.asarray([done for _ in observations], dtype=np.float32),
                    done,
                    np.stack([obs["action_mask"] for obs in observations]),
                    np.stack([obs["action_mask"] for obs in next_observations]),
                )
                if len(self.buffer) >= max(
                    int(self.training_cfg["warmup_steps"]),
                    int(self.training_cfg["batch_size"]),
                ):
                    batch = self.buffer.sample(int(self.training_cfg["batch_size"]))
                    losses.append(self.agent.train_step(batch))

            total_reward += team_reward
            statuses = info.get("shield_statuses", [])
            trace.coverage_curve.append(float(info.get("coverage_ratio", 0.0)))
            trace.team_new_coverage.append(int(info.get("team_new_coverage", 0)))
            trace.repeated_coverage_ratio.append(float(info.get("repeated_coverage_ratio", 0.0)))
            trace.communication_links.append(int(info.get("communication_links", 0)))
            trace.physical_communication_links.append(int(info.get("physical_communication_links", 0)))
            trace.effective_communication_links.append(int(info.get("effective_communication_links", 0)))
            trace.global_sync_applied.append(bool(info.get("global_sync_applied", False)))
            trace.collision_count.append(int(info.get("collision_count", 0)))
            trace.frontier_counts.append(int(sum(info.get("frontier_counts", []))))
            trace.hover_counts.append(sum(status == "hover" for status in statuses))
            trace.hover_reasons.append(list(info.get("hover_reasons", [])))
            trace.goal_conflict_resolutions.append(int(info.get("goal_conflict_resolutions", 0)))
            trace.late_reassignment_applied.append(bool(info.get("late_reassignment_applied", False)))
            trace.adjusted_counts.append(sum(status == "adjusted" for status in statuses))
            trace.planner_failure_counts.append(int(info.get("planner_failure_count", 0)))
            trace.zero_gain_streaks.append(int(info.get("zero_gain_streak", 0)))
            trace.planner_statuses.append(list(info.get("planner_statuses", [])))
            trace.shield_statuses.append(list(statuses))
            trace.active_flags.append(list(info.get("active_flags", [])))
            if replay is not None:
                replay.frames.append(capture_episode_frame(self.env))
            observations = next_observations

        self.last_trace = trace
        self.last_replay = replay
        return EpisodeSummary(
            reward=total_reward,
            coverage_ratio=float(info.get("coverage_ratio", 0.0)),
            episode_length=int(info.get("episode_length", 0)),
            success=bool(info.get("success", False)),
            average_loss=float(np.mean(losses)) if losses else 0.0,
            collision_count=int(info.get("collision_count", 0)),
            repeated_coverage_ratio=float(info.get("repeated_coverage_ratio", 0.0)),
            truncated=bool(info.get("truncated", False)),
            termination_reason=str(info.get("termination_reason", "unknown")),
            mean_acceleration=float(info.get("mean_acceleration", 0.0)),
            max_acceleration=float(info.get("max_acceleration", 0.0)),
            smoothness_cost=float(info.get("smoothness_cost", 0.0)),
        )

    def train(self, num_episodes: int | None = None) -> list[EpisodeSummary]:
        episodes = int(num_episodes or self.training_cfg["num_episodes"])
        if self.num_envs > 1:
            return self._train_parallel(episodes)
        history: list[EpisodeSummary] = []
        checkpoint_dir = Path(self.training_cfg["checkpoint_dir"])
        save_interval = int(self.training_cfg["save_interval"])
        for episode in range(1, episodes + 1):
            summary = self.run_episode(training=True, seed=self.base_seed + episode - 1)
            history.append(summary)
            if save_interval > 0 and episode % save_interval == 0:
                self.agent.save(checkpoint_dir / f"qmix_ep{episode}.pt")
            # Progress bar
            pct = episode / episodes
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "#" * filled + "." * (bar_len - filled)
            loss_str = f"loss={summary.average_loss:.4f}" if summary.average_loss is not None else "loss=n/a"
            print(
                f"\r[{bar}] {episode}/{episodes} | reward={summary.reward:7.2f} "
                f"coverage={summary.coverage_ratio:.3f} {summary.coverage_ratio*100:5.1f}% | "
                f"{loss_str} | {summary.termination_reason}",
                end="", flush=True
            )
        print()  # newline after progress bar
        return history

    def _train_parallel(self, episodes: int) -> list[EpisodeSummary]:
        history: list[EpisodeSummary] = []
        checkpoint_dir = Path(self.training_cfg["checkpoint_dir"])
        save_interval = int(self.training_cfg["save_interval"])
        with ParallelRolloutManager(self.config, self.num_envs) as manager:
            ready = manager.start()
            observations = [item.observations for item in ready]
            states = [item.state for item in ready]
            totals = [0.0 for _ in range(self.num_envs)]
            losses = [[] for _ in range(self.num_envs)]
            traces = [self._new_trace(item.reset_info) for item in ready]

            while len(history) < episodes:
                action_batches = self.agent.select_actions_batch(observations, explore=True)
                steps = manager.step(action_batches)
                for worker_index, step in enumerate(steps):
                    current_observations = observations[worker_index]
                    done = step.terminated or step.truncated
                    team_reward = float(np.sum(step.rewards))
                    self.buffer.add(
                        np.stack([flatten_observation(obs) for obs in current_observations]),
                        np.asarray(action_batches[worker_index], dtype=np.int64),
                        states[worker_index],
                        team_reward,
                        np.stack([flatten_observation(obs) for obs in step.next_observations]),
                        step.next_state,
                        np.asarray([done for _ in current_observations], dtype=np.float32),
                        done,
                        np.stack([obs["action_mask"] for obs in current_observations]),
                        np.stack([obs["action_mask"] for obs in step.next_observations]),
                    )
                    if len(self.buffer) >= max(
                        int(self.training_cfg["warmup_steps"]),
                        int(self.training_cfg["batch_size"]),
                    ):
                        batch = self.buffer.sample(int(self.training_cfg["batch_size"]))
                        losses[worker_index].append(self.agent.train_step(batch))

                    totals[worker_index] += team_reward
                    self._append_trace(traces[worker_index], step.info)

                    if done and len(history) < episodes:
                        summary = self._summary_from_info(
                            total_reward=totals[worker_index],
                            losses=losses[worker_index],
                            info=step.info,
                        )
                        history.append(summary)
                        self.last_trace = traces[worker_index]
                        if save_interval > 0 and len(history) % save_interval == 0:
                            self.agent.save(checkpoint_dir / f"qmix_ep{len(history)}.pt")
                        self._print_progress(len(history), episodes, summary)

                    if done:
                        observations[worker_index] = step.reset_observations or []
                        states[worker_index] = (
                            step.reset_state
                            if step.reset_state is not None
                            else np.zeros_like(states[worker_index])
                        )
                        totals[worker_index] = 0.0
                        losses[worker_index] = []
                        traces[worker_index] = self._new_trace(step.reset_info or step.info)
                    else:
                        observations[worker_index] = step.next_observations
                        states[worker_index] = step.next_state
        print()
        return history

    @staticmethod
    def _new_trace(reset_info: dict[str, Any]) -> EpisodeTrace:
        return EpisodeTrace(
            coverage_curve=[float(reset_info["coverage_ratio"])],
            team_new_coverage=[0],
            repeated_coverage_ratio=[0.0],
            communication_links=[int(reset_info["communication_links"])],
            physical_communication_links=[int(reset_info["physical_communication_links"])],
            effective_communication_links=[int(reset_info["effective_communication_links"])],
            global_sync_applied=[bool(reset_info["global_sync_applied"])],
            collision_count=[0],
            frontier_counts=[int(sum(reset_info["frontier_counts"]))],
            hover_counts=[0],
            hover_reasons=[list(reset_info["hover_reasons"])],
            goal_conflict_resolutions=[int(reset_info["goal_conflict_resolutions"])],
            late_reassignment_applied=[bool(reset_info["late_reassignment_applied"])],
            adjusted_counts=[0],
            planner_failure_counts=[int(reset_info["planner_failure_count"])],
            zero_gain_streaks=[int(reset_info["zero_gain_streak"])],
            planner_statuses=[list(reset_info["planner_statuses"])],
            shield_statuses=[list(reset_info["shield_statuses"])],
            active_flags=[list(reset_info["active_flags"])],
        )

    @staticmethod
    def _append_trace(trace: EpisodeTrace, info: dict[str, Any]) -> None:
        statuses = info.get("shield_statuses", [])
        trace.coverage_curve.append(float(info.get("coverage_ratio", 0.0)))
        trace.team_new_coverage.append(int(info.get("team_new_coverage", 0)))
        trace.repeated_coverage_ratio.append(float(info.get("repeated_coverage_ratio", 0.0)))
        trace.communication_links.append(int(info.get("communication_links", 0)))
        trace.physical_communication_links.append(int(info.get("physical_communication_links", 0)))
        trace.effective_communication_links.append(int(info.get("effective_communication_links", 0)))
        trace.global_sync_applied.append(bool(info.get("global_sync_applied", False)))
        trace.collision_count.append(int(info.get("collision_count", 0)))
        trace.frontier_counts.append(int(sum(info.get("frontier_counts", []))))
        trace.hover_counts.append(sum(status == "hover" for status in statuses))
        trace.hover_reasons.append(list(info.get("hover_reasons", [])))
        trace.goal_conflict_resolutions.append(int(info.get("goal_conflict_resolutions", 0)))
        trace.late_reassignment_applied.append(bool(info.get("late_reassignment_applied", False)))
        trace.adjusted_counts.append(sum(status == "adjusted" for status in statuses))
        trace.planner_failure_counts.append(int(info.get("planner_failure_count", 0)))
        trace.zero_gain_streaks.append(int(info.get("zero_gain_streak", 0)))
        trace.planner_statuses.append(list(info.get("planner_statuses", [])))
        trace.shield_statuses.append(list(statuses))
        trace.active_flags.append(list(info.get("active_flags", [])))

    @staticmethod
    def _summary_from_info(total_reward: float, losses: list[float], info: dict[str, Any]) -> EpisodeSummary:
        return EpisodeSummary(
            reward=total_reward,
            coverage_ratio=float(info.get("coverage_ratio", 0.0)),
            episode_length=int(info.get("episode_length", 0)),
            success=bool(info.get("success", False)),
            average_loss=float(np.mean(losses)) if losses else 0.0,
            collision_count=int(info.get("collision_count", 0)),
            repeated_coverage_ratio=float(info.get("repeated_coverage_ratio", 0.0)),
            truncated=bool(info.get("truncated", False)),
            termination_reason=str(info.get("termination_reason", "unknown")),
            mean_acceleration=float(info.get("mean_acceleration", 0.0)),
            max_acceleration=float(info.get("max_acceleration", 0.0)),
            smoothness_cost=float(info.get("smoothness_cost", 0.0)),
        )

    @staticmethod
    def _print_progress(episode: int, episodes: int, summary: EpisodeSummary) -> None:
        pct = episode / episodes
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "#" * filled + "." * (bar_len - filled)
        loss_str = f"loss={summary.average_loss:.4f}" if summary.average_loss is not None else "loss=n/a"
        print(
            f"\r[{bar}] {episode}/{episodes} | reward={summary.reward:7.2f} "
            f"coverage={summary.coverage_ratio:.3f} {summary.coverage_ratio*100:5.1f}% | "
            f"{loss_str} | {summary.termination_reason}",
            end="",
            flush=True,
        )

    def evaluate(
        self,
        episodes: int = 5,
        capture_replay: bool = True,
        seeds: list[int] | None = None,
    ) -> list[EpisodeSummary]:
        history: list[EpisodeSummary] = []
        self.evaluation_traces = []
        self.evaluation_replays = []
        evaluation_seeds = seeds if seeds is not None else [None for _ in range(episodes)]
        for seed in evaluation_seeds:
            history.append(self.run_episode(training=False, capture_replay=capture_replay, seed=seed))
            if self.last_trace is not None:
                self.evaluation_traces.append(self.last_trace)
            if self.last_replay is not None:
                self.evaluation_replays.append(self.last_replay)
        return history
