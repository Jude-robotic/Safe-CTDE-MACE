from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import distance_transform_edt

from safe_ctde_mace.agents.uav_agent import UAVAgent
from safe_ctde_mace.communication.comm_graph import CommGraph
from safe_ctde_mace.communication.map_fusion import MapFusion
from safe_ctde_mace.envs.voxel_world import VoxelWorld
from safe_ctde_mace.mapping.coverage_map import CoverageMap, SensorUpdate
from safe_ctde_mace.mapping.frontier_detector import CandidateSet, FrontierDetector
from safe_ctde_mace.mapping.voxel_map import VoxelState, encode_state_channels
from safe_ctde_mace.planning.astar_3d import AStar3D
from safe_ctde_mace.planning.ego_planner import ContinuousTrajectory, EGOStylePlanner
from safe_ctde_mace.planning.safety_shield import ShieldResult, SafetyShield
from safe_ctde_mace.planning.trajectory_tracker import TrajectoryTracker
from safe_ctde_mace.utils.geometry import euclidean_distance
from safe_ctde_mace.utils.metrics import repeated_coverage_ratio, trajectory_metrics_from_points


class MultiUAVCoverageEnv:
    """Gymnasium-like multi-UAV coverage exploration environment."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = deepcopy(config)
        self.env_cfg = self.config["environment"]
        self.reward_cfg = self.config["reward"]
        self.seed = int(self.config.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        self.world: VoxelWorld | None = None
        self.global_coverage: CoverageMap | None = None
        self.agents: list[UAVAgent] = []
        self.comm_graph = CommGraph(self.env_cfg["comm_range"])
        self.frontier_detector = FrontierDetector(
            num_candidates=self.env_cfg["num_frontier_candidates"],
            sensor_range=self.env_cfg["sensor_range"],
            reservation_radius=self.env_cfg["reservation_radius"],
            candidate_min_separation=float(self.env_cfg.get("candidate_min_separation", 0.0)),
            max_neighbors=int(self.env_cfg.get("max_neighbors", max(self.env_cfg["num_uavs"] - 1, 0))),
        )
        self.planner = AStar3D(self.env_cfg["astar_connectivity"])
        self.motion_planner_type = str(self.env_cfg.get("planner_type", "astar"))
        self.ego_planner = EGOStylePlanner(
            max_velocity=self.env_cfg["max_velocity"],
            max_acceleration=self.env_cfg["max_acceleration"],
            safe_obs_dist=self.env_cfg["safe_obs_dist"],
            sample_dt=float(self.env_cfg.get("trajectory_sample_dt", 0.5)),
            optimize_iterations=int(self.env_cfg.get("ego_optimize_iterations", 30)),
            smooth_weight=float(self.env_cfg.get("ego_smooth_weight", 0.35)),
            obstacle_weight=float(self.env_cfg.get("ego_obstacle_weight", 0.8)),
            seed_connectivity=int(self.env_cfg.get("ego_seed_connectivity", 26)),
        )
        self.safety_shield = SafetyShield(
            self.env_cfg["safe_obs_dist"],
            self.env_cfg["safe_agent_dist"],
        )
        self.tracker = TrajectoryTracker()

        self.step_count = 0
        self.adjacency = np.zeros((self.env_cfg["num_uavs"], self.env_cfg["num_uavs"]), dtype=bool)
        self.physical_adjacency = self.adjacency.copy()
        self.global_sync_applied = False
        self.neighbor_lists: list[list[int]] = [[] for _ in range(self.env_cfg["num_uavs"])]
        self.current_candidates: list[CandidateSet] = []
        self.frontier_sets: list[set[tuple[int, int, int]]] = []
        self.local_state_arrays: list[np.ndarray] = []
        self.obstacle_distance_fields: list[np.ndarray] = []
        self.zero_gain_streak = 0
        self.recent_team_gains: list[int] = []
        self.late_reassignment_applied = False
        self.last_info: dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

        self.world = self._build_world()
        self.global_coverage = CoverageMap(self.world.grid_size)
        obstacle_indices = [tuple(index) for index in np.argwhere(self.world.grid == int(VoxelState.OBSTACLE))]
        self.global_coverage.mark_obstacle(obstacle_indices)
        self.agents = [
            UAVAgent.create(
                agent_id=index,
                position=position,
                sensor_range=self.env_cfg["sensor_range"],
                map_shape=self.world.grid_size,
            )
            for index, position in enumerate(self._initial_positions())
        ]
        self.step_count = 0
        self.zero_gain_streak = 0
        self.recent_team_gains = []
        self.late_reassignment_applied = False
        self._observe_all_agents()
        self._refresh_communication_and_candidates()
        observations = self.get_obs()
        info = self._build_info(
            new_coverage=[0 for _ in self.agents],
            repeated_coverage=[0 for _ in self.agents],
            unknown_reduction=[0 for _ in self.agents],
            collision_flags=[False for _ in self.agents],
            obstacle_collision_flags=[False for _ in self.agents],
            inter_uav_collision_flags=[False for _ in self.agents],
            path_lengths=[0.0 for _ in self.agents],
            mean_accelerations=[0.0 for _ in self.agents],
            max_accelerations=[0.0 for _ in self.agents],
            smoothness_costs=[0.0 for _ in self.agents],
            team_new_coverage=0,
            terminated=False,
            truncated=False,
            planner_statuses=["reset" for _ in self.agents],
            shield_statuses=["reset" for _ in self.agents],
            hover_reasons=["" for _ in self.agents],
            goal_conflict_resolutions=0,
            late_reassignment_applied=False,
            active_flags=[agent.active for agent in self.agents],
        )
        self.last_info = info
        return observations, info

    def step(
        self,
        actions: Iterable[int],
    ) -> tuple[list[dict[str, np.ndarray]], list[float], bool, bool, dict[str, Any]]:
        if self.world is None or self.global_coverage is None:
            raise RuntimeError("Environment must be reset before stepping.")

        action_list = [int(action) for action in actions]
        if len(action_list) != len(self.agents):
            raise ValueError("Expected one action per UAV.")

        selected_actions = [self._sanitize_action(index, action) for index, action in enumerate(action_list)]
        self.late_reassignment_applied = self._should_apply_late_reassignment()
        if self.late_reassignment_applied:
            selected_actions = self._late_reassign_actions(selected_actions)
        selected_actions, goal_conflict_resolutions = self._deconflict_selected_actions(selected_actions)
        reserved_penalties = [
            float(self.current_candidates[index].features[action, 3]) if action >= 0 else 0.0
            for index, action in enumerate(selected_actions)
        ]

        self._broadcast_reservations(selected_actions)
        self._refresh_local_state_arrays()

        shield_results: list[ShieldResult] = []
        path_lengths: list[float] = []
        for index, agent in enumerate(self.agents):
            candidate_set = self.current_candidates[index]
            if not agent.active or selected_actions[index] < 0:
                hover_reason = "inactive_agent" if not agent.active else "no_valid_candidate"
                shield_results.append(
                    ShieldResult(agent.current_voxel, [agent.current_voxel], None, "hover", hover_reason)
                )
                path_lengths.append(0.0)
                continue

            result = self.safety_shield.select_safe_goal(
                current_position=agent.current_voxel,
                candidate_goals=candidate_set.goals,
                action_mask=candidate_set.action_mask,
                chosen_action=selected_actions[index],
                knowledge_states=self.local_state_arrays[index],
                neighbor_states=self._neighbor_state_dicts(index),
                planner=self.planner,
                obstacle_distance=self.obstacle_distance_fields[index],
            )
            shield_results.append(result)
            agent.set_goal(result.safe_goal)
            path_lengths.append(max(len(result.path) - 1, 0))

        shield_results = self._resolve_step_conflicts(shield_results)
        execution_plans: list[list[tuple[int, int, int]] | ContinuousTrajectory] = []
        trajectory_lengths: list[float] = []
        mean_accelerations: list[float] = []
        max_accelerations: list[float] = []
        smoothness_costs: list[float] = []
        planner_statuses: list[str] = []
        for index, (agent, result) in enumerate(zip(self.agents, shield_results, strict=True)):
            if not agent.active or result.status == "hover":
                execution_plans.append(result.path)
                trajectory_lengths.append(0.0)
                mean_accelerations.append(0.0)
                max_accelerations.append(0.0)
                smoothness_costs.append(0.0)
                planner_statuses.append("not_requested")
                continue
            if self.motion_planner_type == "ego":
                planner_result = self.ego_planner.plan_with_status(
                    agent.position,
                    result.safe_goal,
                    self.local_state_arrays[index],
                )
                trajectory = planner_result.trajectory
                if trajectory is None:
                    shield_results[index] = ShieldResult(
                        agent.current_voxel,
                        [agent.current_voxel],
                        None,
                        "hover",
                        "planner_unavailable",
                    )
                    execution_plans.append([agent.current_voxel])
                    trajectory_lengths.append(0.0)
                    mean_accelerations.append(0.0)
                    max_accelerations.append(0.0)
                    smoothness_costs.append(0.0)
                    planner_statuses.append(planner_result.status)
                    continue
                metrics = trajectory.metrics()
                execution_plans.append(trajectory)
                trajectory_lengths.append(metrics["path_length"])
                mean_accelerations.append(metrics["mean_acceleration"])
                max_accelerations.append(metrics["max_acceleration"])
                smoothness_costs.append(metrics["smoothness_cost"])
                planner_statuses.append(planner_result.status)
            else:
                execution_plans.append(result.path)
                metrics = trajectory_metrics_from_points(np.asarray(result.path, dtype=float))
                trajectory_lengths.append(metrics["path_length"])
                mean_accelerations.append(metrics["mean_acceleration"])
                max_accelerations.append(metrics["max_acceleration"])
                smoothness_costs.append(metrics["smoothness_cost"])
                planner_statuses.append("planned")

        previous_positions = [agent.position.copy() for agent in self.agents]
        proposed_positions: list[np.ndarray] = []
        proposed_velocities: list[np.ndarray] = []
        step_distances: list[float] = []
        sampled_segments: list[np.ndarray] = []
        for agent, plan in zip(self.agents, execution_plans, strict=True):
            if self.motion_planner_type == "ego" and isinstance(plan, ContinuousTrajectory):
                next_position, velocity, step_distance, segment, _ = self.tracker.step_continuous(
                    agent.position,
                    plan,
                    float(self.env_cfg.get("trajectory_execution_dt", 1.0)),
                )
            else:
                next_position, velocity, step_distance = self.tracker.step(agent.current_voxel, plan)
                segment = np.stack([agent.position.copy(), np.asarray(next_position, dtype=float)])
            proposed_positions.append(np.asarray(next_position, dtype=float))
            proposed_velocities.append(np.asarray(velocity, dtype=float))
            step_distances.append(step_distance)
            sampled_segments.append(segment)

        collision_flags, obstacle_collision_flags, inter_uav_collision_flags = self._detect_collisions(
            sampled_segments,
            proposed_positions,
        )
        self._apply_motion_with_soft_collision_recovery(
            previous_positions,
            proposed_positions,
            proposed_velocities,
            collision_flags,
        )
        local_updates, global_updates = self._observe_all_agents()
        team_new_coverage = int(sum(update.new_covered for update in global_updates))
        self.zero_gain_streak = self.zero_gain_streak + 1 if team_new_coverage == 0 else 0
        self._append_recent_team_gain(team_new_coverage)

        rewards = self.compute_reward(
            local_updates=local_updates,
            global_updates=global_updates,
            collision_flags=collision_flags,
            obstacle_collision_flags=obstacle_collision_flags,
            inter_uav_collision_flags=inter_uav_collision_flags,
            step_distances=step_distances,
            reserved_penalties=reserved_penalties,
            team_new_coverage=team_new_coverage,
            selected_actions=selected_actions,
        )

        self.step_count += 1
        terminated = self.is_done()
        truncated = self.step_count >= self.env_cfg["max_steps"]
        self._refresh_communication_and_candidates()
        observations = self.get_obs()
        info = self._build_info(
            new_coverage=[update.new_covered for update in global_updates],
            repeated_coverage=[update.repeated_covered for update in global_updates],
            unknown_reduction=[update.unknown_reduction for update in local_updates],
            collision_flags=collision_flags,
            obstacle_collision_flags=obstacle_collision_flags,
            inter_uav_collision_flags=inter_uav_collision_flags,
            path_lengths=trajectory_lengths,
            mean_accelerations=mean_accelerations,
            max_accelerations=max_accelerations,
            smoothness_costs=smoothness_costs,
            team_new_coverage=team_new_coverage,
            terminated=terminated,
            truncated=truncated,
            planner_statuses=planner_statuses,
            shield_statuses=[result.status for result in shield_results],
            hover_reasons=[result.hover_reason or "" for result in shield_results],
            goal_conflict_resolutions=goal_conflict_resolutions,
            late_reassignment_applied=self.late_reassignment_applied,
            active_flags=[agent.active for agent in self.agents],
        )
        self.last_info = info
        return observations, rewards, terminated, truncated, info

    def get_obs(self) -> list[dict[str, np.ndarray]]:
        observations: list[dict[str, np.ndarray]] = []
        for index, agent in enumerate(self.agents):
            patch = agent.local_map.get_patch(
                agent.current_voxel,
                self.env_cfg["local_patch_radius"],
                states=self.local_state_arrays[index],
            )
            channels = encode_state_channels(patch)
            neighbor_occupancy = self._neighbor_occupancy_patch(index, patch.shape)
            distance_field = self._obstacle_distance_patch(patch)
            local_voxel_map = np.concatenate(
                [channels, neighbor_occupancy[None, ...], distance_field[None, ...]],
                axis=0,
            )

            goal = agent.goal if agent.goal is not None else agent.position
            self_state = np.concatenate([agent.position, agent.velocity, goal]).astype(np.float32)
            coverage_ratio = np.asarray([self.global_coverage.coverage_ratio(self.world)], dtype=np.float32)
            observations.append(
                {
                    "local_voxel_map": local_voxel_map.astype(np.float32),
                    "self_state": self_state,
                    "neighbor_states": self._neighbor_summary(index),
                    "coverage_ratio": coverage_ratio,
                    "candidate_features": self.current_candidates[index].features.copy(),
                    "action_mask": self.current_candidates[index].action_mask.copy(),
                }
            )
        return observations

    def get_global_state(self) -> np.ndarray:
        """Return a compact centralized state vector used only during training."""
        if self.world is None or self.global_coverage is None:
            raise RuntimeError("Environment must be reset before reading the global state.")

        shape_scale = np.maximum(np.asarray(self.world.grid_size, dtype=np.float32) - 1.0, 1.0)
        total_voxels = float(np.prod(self.world.grid_size))
        state_counts = np.bincount(
            self.global_coverage.base_states.reshape(-1),
            minlength=int(VoxelState.RESERVED) + 1,
        ).astype(np.float32)
        count_features = state_counts[: int(VoxelState.COVERED) + 1] / max(total_voxels, 1.0)
        coverage_ratio = np.asarray([self.global_coverage.coverage_ratio(self.world)], dtype=np.float32)

        agent_features: list[np.ndarray] = []
        for agent in self.agents:
            goal = agent.goal if agent.goal is not None else agent.position
            agent_features.append(
                np.concatenate(
                    [
                        agent.position.astype(np.float32) / shape_scale,
                        agent.velocity.astype(np.float32) / max(float(self.env_cfg["max_velocity"]), 1e-6),
                        goal.astype(np.float32) / shape_scale,
                    ]
                )
            )
        adjacency = self.adjacency.astype(np.float32).reshape(-1)
        frontier_features = np.asarray(
            [len(frontiers) / max(total_voxels, 1.0) for frontiers in self.frontier_sets],
            dtype=np.float32,
        )
        valid_action_features = np.asarray(
            [
                np.count_nonzero(candidates.action_mask) / max(len(candidates.action_mask), 1)
                for candidates in self.current_candidates
            ],
            dtype=np.float32,
        )
        residual_coverage_features = self._residual_coverage_distribution()
        return np.concatenate(
            [
                coverage_ratio,
                count_features,
                residual_coverage_features,
                *agent_features,
                adjacency,
                frontier_features,
                valid_action_features,
            ]
        ).astype(np.float32)

    def compute_reward(
        self,
        local_updates: list[SensorUpdate],
        global_updates: list[SensorUpdate],
        collision_flags: list[bool],
        obstacle_collision_flags: list[bool],
        inter_uav_collision_flags: list[bool],
        step_distances: list[float],
        reserved_penalties: list[float],
        team_new_coverage: int,
        selected_actions: list[int] | None = None,
    ) -> list[float]:
        rewards: list[float] = []
        team_observed = max(sum(len(update.observed_indices) for update in global_updates), 1)
        team_share = team_new_coverage / team_observed
        overlap_values = self._coverage_overlap_penalties()
        coverage = self.global_coverage.coverage_ratio(self.world)
        finish_bonus = (
            self.reward_cfg["w_finish"]
            if coverage >= self.env_cfg["target_coverage_ratio"]
            else 0.0
        )
        # Milestone bonuses for intermediate coverage targets
        milestone_bonus = 0.0
        if coverage >= 0.50:
            milestone_bonus += 2.0
        if coverage >= 0.70:
            milestone_bonus += 4.0
        if coverage >= 0.85:
            milestone_bonus += 8.0
        # Dense coverage progress reward - scaled to target to avoid scale explosion
        progress_weight = float(self.reward_cfg.get("w_progress", 0.0))
        target = float(self.env_cfg["target_coverage_ratio"])
        coverage_progress = coverage / target if target > 0 else 0.0
        progress_reward = progress_weight * coverage_progress
        exploration_bonus = self._compute_exploration_bonus(selected_actions)

        for index, _ in enumerate(self.agents):
            local_observed = max(len(local_updates[index].observed_indices), 1)
            global_observed = max(len(global_updates[index].observed_indices), 1)
            normalized_new = global_updates[index].new_covered / global_observed
            normalized_info = local_updates[index].unknown_reduction / local_observed
            normalized_repeat = global_updates[index].repeated_covered / global_observed
            blended_new = 0.7 * normalized_new + 0.3 * team_share
            reward = (
                self.reward_cfg["w_new"] * blended_new
                + self.reward_cfg["w_info"] * normalized_info
                - self.reward_cfg["w_repeat"] * normalized_repeat
                - self.reward_cfg["w_overlap"] * overlap_values[index]
                - self.reward_cfg["w_collision"] * float(inter_uav_collision_flags[index])
                - self.reward_cfg["w_obstacle"] * float(obstacle_collision_flags[index])
                - self.reward_cfg["w_time"]
                - self.reward_cfg["w_energy"] * step_distances[index]
                - self.reward_cfg["w_reserve"] * reserved_penalties[index]
                + progress_reward
                + finish_bonus
                + milestone_bonus
                + exploration_bonus[index]
            )
            rewards.append(float(reward))
        return rewards

    def is_done(self) -> bool:
        if self.world is None or self.global_coverage is None:
            return False
        coverage_complete = self.global_coverage.coverage_ratio(self.world) >= self.env_cfg[
            "target_coverage_ratio"
        ]
        all_failed = all(not agent.active for agent in self.agents)
        return bool(coverage_complete or all_failed)

    def render(self):
        from safe_ctde_mace.utils.visualization import plot_episode

        return plot_episode(self)

    def _build_world(self) -> VoxelWorld:
        world = VoxelWorld(
            grid_size=self.env_cfg["grid_size"],
            voxel_resolution=self.env_cfg["voxel_resolution"],
            seed=self.seed,
        )
        obstacle_cfg = self.env_cfg.get("obstacle_generation", {})
        for box in obstacle_cfg.get("manual_boxes", []):
            world.add_box(box["min_corner"], box["max_corner"])
        world.add_random_obstacles(
            count=int(obstacle_cfg.get("random_boxes", 0)),
            min_box_size=obstacle_cfg.get("min_box_size", [1, 1, 1]),
            max_box_size=obstacle_cfg.get("max_box_size", [1, 1, 1]),
            forbidden_positions=self._initial_positions(),
        )
        return world

    def _initial_positions(self) -> list[tuple[int, int, int]]:
        configured = self.env_cfg.get("initial_positions")
        if configured:
            return [tuple(int(value) for value in position) for position in configured]
        if self.world is None:
            world_shape = tuple(int(value) for value in self.env_cfg["grid_size"])
        else:
            world_shape = self.world.grid_size
        positions: list[tuple[int, int, int]] = []
        while len(positions) < self.env_cfg["num_uavs"]:
            candidate = tuple(
                int(self.rng.integers(0, limit))
                for limit in world_shape
            )
            if candidate not in positions:
                positions.append(candidate)
        return positions

    def _observe_all_agents(self) -> tuple[list[SensorUpdate], list[SensorUpdate]]:
        if self.world is None or self.global_coverage is None:
            raise RuntimeError("World and coverage map must be initialized.")

        local_updates: list[SensorUpdate] = []
        global_updates: list[SensorUpdate] = []
        for agent in self.agents:
            local_updates.append(agent.observe(self.world))
            global_updates.append(
                self.global_coverage.update_from_sensor(
                    self.world,
                    agent.current_voxel,
                    self.env_cfg["sensor_range"],
                )
            )
        return local_updates, global_updates

    def _refresh_communication_and_candidates(self) -> None:
        positions = [agent.position for agent in self.agents]
        self.physical_adjacency = self.comm_graph.adjacency_matrix(positions)
        self.global_sync_applied = self._should_apply_global_sync()
        self.adjacency = (
            self._fully_connected_adjacency(len(self.agents))
            if self.global_sync_applied
            else self.physical_adjacency.copy()
        )
        self.neighbor_lists = [list(np.flatnonzero(row)) for row in self.adjacency]
        MapFusion.fuse_neighbors([agent.local_map for agent in self.agents], self.neighbor_lists)
        self.current_candidates = []
        self.frontier_sets = []
        self._refresh_local_state_arrays()
        for index, agent in enumerate(self.agents):
            neighbor_positions = [self.agents[neighbor].position for neighbor in self._ordered_neighbor_indices(index)]
            candidates = self.frontier_detector.generate_candidates(
                self.local_state_arrays[index],
                agent.current_voxel,
                neighbor_positions,
            )
            self.current_candidates.append(candidates)
            self.frontier_sets.append(candidates.frontier_voxels)

    def _refresh_local_state_arrays(self) -> None:
        self.local_state_arrays = [agent.local_map.as_array() for agent in self.agents]
        self.obstacle_distance_fields = [
            distance_transform_edt(states != int(VoxelState.OBSTACLE))
            for states in self.local_state_arrays
        ]

    def _broadcast_reservations(self, selected_actions: list[int]) -> None:
        for agent in self.agents:
            agent.local_map.clear_reserved()
            agent.clear_reservation()
        for index, action in enumerate(selected_actions):
            if action < 0:
                continue
            goal = self.current_candidates[index].goals[action]
            self.agents[index].reserve_region_around(goal, self.env_cfg["reservation_radius"])
        MapFusion.fuse_neighbors([agent.local_map for agent in self.agents], self.neighbor_lists)

    def _should_apply_global_sync(self) -> bool:
        interval = int(self.env_cfg.get("global_sync_interval", 0))
        return interval > 0 and self.step_count % interval == 0

    def _should_apply_late_reassignment(self) -> bool:
        if not bool(self.env_cfg.get("late_reassign_enabled", False)):
            return False
        coverage_ready = self.global_coverage.coverage_ratio(self.world) > float(
            self.env_cfg.get("late_reassign_min_coverage", 0.70)
        )
        zero_gain_ready = self.zero_gain_streak >= int(self.env_cfg.get("late_reassign_zero_gain_streak", 5))
        low_gain_ready = self._recent_gain_window_is_low()
        return bool(coverage_ready and (zero_gain_ready or low_gain_ready))

    @staticmethod
    def _fully_connected_adjacency(count: int) -> np.ndarray:
        adjacency = np.ones((count, count), dtype=bool)
        np.fill_diagonal(adjacency, False)
        return adjacency

    def _sanitize_action(self, agent_index: int, action: int) -> int:
        mask = self.current_candidates[agent_index].action_mask
        valid_actions = np.flatnonzero(mask)
        if len(valid_actions) == 0:
            return -1
        if 0 <= action < len(mask) and mask[action]:
            return int(action)
        return int(valid_actions[0])

    def _late_reassign_actions(self, fallback_actions: list[int]) -> list[int]:
        reassigned = [-1 for _ in self.agents]
        layout = self.frontier_detector.feature_layout
        ranked_options: list[tuple[float, float, float, int, int, tuple[int, int, int]]] = []
        for agent_index, agent in enumerate(self.agents):
            if not agent.active:
                continue
            candidates = self.current_candidates[agent_index]
            for action in np.flatnonzero(candidates.action_mask):
                action_index = int(action)
                goal = tuple(int(value) for value in candidates.goals[action_index])
                path_cost = float(candidates.features[action_index, 5])
                info_gain = float(candidates.features[action_index, 1])
                uncovered_density = float(candidates.features[action_index, layout.uncovered_density])
                ranked_options.append((path_cost, -uncovered_density, -info_gain, agent_index, action_index, goal))

        assigned_agents: set[int] = set()
        assigned_goals: set[tuple[int, int, int]] = set()
        for _, _, _, agent_index, action_index, goal in sorted(ranked_options):
            if agent_index in assigned_agents or goal in assigned_goals:
                continue
            reassigned[agent_index] = action_index
            assigned_agents.add(agent_index)
            assigned_goals.add(goal)

        for agent_index, action in enumerate(fallback_actions):
            if reassigned[agent_index] >= 0:
                continue
            reassigned[agent_index] = action
        return reassigned

    def _deconflict_selected_actions(self, selected_actions: list[int]) -> tuple[list[int], int]:
        resolved = list(selected_actions)
        accepted: list[tuple[int, np.ndarray]] = []
        resolution_count = 0
        min_goal_separation = max(2.0 * float(self.env_cfg["reservation_radius"]), 1.0)
        for index, action in enumerate(selected_actions):
            if action < 0:
                continue
            candidates = self.current_candidates[index]
            valid_actions = [int(action)] + [
                int(candidate)
                for candidate in np.flatnonzero(candidates.action_mask)
                if int(candidate) != action
            ]
            chosen_action = int(action)
            for candidate_action in valid_actions:
                goal = np.asarray(candidates.goals[candidate_action], dtype=float)
                neighbor_goals = [
                    accepted_goal
                    for accepted_index, accepted_goal in accepted
                    if bool(self.adjacency[index, accepted_index])
                ]
                if all(
                    euclidean_distance(goal, accepted_goal) >= min_goal_separation
                    for accepted_goal in neighbor_goals
                ):
                    chosen_action = candidate_action
                    break
            if chosen_action != action:
                resolution_count += 1
            resolved[index] = chosen_action
            accepted.append((index, np.asarray(candidates.goals[chosen_action], dtype=float)))
        return resolved, resolution_count

    def _neighbor_state_dicts(self, agent_index: int) -> list[dict[str, np.ndarray]]:
        neighbors = []
        for neighbor_index in self.neighbor_lists[agent_index]:
            neighbor = self.agents[neighbor_index]
            neighbors.append(
                {
                    "position": neighbor.position.copy(),
                    "velocity": neighbor.velocity.copy(),
                }
            )
        return neighbors

    def _neighbor_summary(self, agent_index: int) -> np.ndarray:
        max_neighbors = int(self.env_cfg.get("max_neighbors", max(len(self.agents) - 1, 1)))
        summary = np.zeros((max_neighbors, 9), dtype=np.float32)
        neighbors = self._ordered_neighbor_indices(agent_index)
        for row, neighbor_index in enumerate(neighbors):
            neighbor = self.agents[neighbor_index]
            goal = neighbor.goal if neighbor.goal is not None else neighbor.position
            summary[row] = np.concatenate([neighbor.position, neighbor.velocity, goal]).astype(np.float32)
        return summary

    def _ordered_neighbor_indices(self, agent_index: int) -> list[int]:
        agent = self.agents[agent_index]
        max_neighbors = int(self.env_cfg.get("max_neighbors", max(len(self.agents) - 1, 1)))
        return sorted(
            self.neighbor_lists[agent_index],
            key=lambda index: euclidean_distance(agent.position, self.agents[index].position),
        )[:max_neighbors]

    def _neighbor_occupancy_patch(self, agent_index: int, patch_shape: tuple[int, int, int]) -> np.ndarray:
        occupancy = np.zeros(patch_shape, dtype=np.float32)
        center = np.asarray(self.agents[agent_index].current_voxel, dtype=int)
        radius = self.env_cfg["local_patch_radius"]
        for neighbor_index in self.neighbor_lists[agent_index]:
            relative = np.asarray(self.agents[neighbor_index].current_voxel, dtype=int) - center + radius
            if np.all(relative >= 0) and np.all(relative < np.asarray(patch_shape)):
                occupancy[tuple(relative)] = 1.0
        return occupancy

    @staticmethod
    def _obstacle_distance_patch(patch: np.ndarray) -> np.ndarray:
        obstacle_mask = patch == int(VoxelState.OBSTACLE)
        if not np.any(obstacle_mask):
            return np.ones_like(patch, dtype=np.float32)
        distances = distance_transform_edt(~obstacle_mask)
        max_distance = max(float(np.max(distances)), 1.0)
        return (distances / max_distance).astype(np.float32)

    def _coverage_overlap_penalties(self) -> list[float]:
        penalties = [0.0 for _ in self.agents]
        for source in range(len(self.agents)):
            for target in range(source + 1, len(self.agents)):
                distance = euclidean_distance(self.agents[source].position, self.agents[target].position)
                overlap = max(
                    0.0,
                    1.0 - distance / max(2.0 * self.env_cfg["sensor_range"], 1e-6),
                )
                penalties[source] += overlap
                penalties[target] += overlap
        return penalties

    def _append_recent_team_gain(self, team_new_coverage: int) -> None:
        window = max(int(self.env_cfg.get("late_reassign_window", 0)), 0)
        if window <= 0:
            return
        self.recent_team_gains.append(int(team_new_coverage))
        if len(self.recent_team_gains) > window:
            self.recent_team_gains = self.recent_team_gains[-window:]

    def _recent_gain_window_is_low(self) -> bool:
        window = max(int(self.env_cfg.get("late_reassign_window", 0)), 0)
        if window <= 0 or len(self.recent_team_gains) < window:
            return False
        threshold = float(self.env_cfg.get("late_reassign_max_mean_gain", 0.0))
        return float(np.mean(self.recent_team_gains[-window:])) <= threshold

    def _compute_exploration_bonus(self, selected_actions: list[int] | None = None) -> list[float]:
        coverage = self.global_coverage.coverage_ratio(self.world)
        if coverage <= 0.40 or self.zero_gain_streak < 3:
            return [0.0 for _ in self.agents]
        layout = self.frontier_detector.feature_layout
        bonuses = []
        for index, _ in enumerate(self.agents):
            candidates = self.current_candidates[index]
            valid_indices = np.flatnonzero(candidates.action_mask)
            if len(valid_indices) == 0:
                bonuses.append(0.0)
                continue
            action = selected_actions[index] if selected_actions is not None else int(valid_indices[0])
            if action not in valid_indices:
                bonuses.append(0.0)
                continue
            feat = candidates.features[int(action)]
            uncovered_density = feat[layout.uncovered_density]
            grid_quadrant = feat[layout.grid_quadrant]
            layer_height = feat[layout.layer_height]
            diversity_score = 0.5 * uncovered_density + 0.1 * (1.0 - abs(grid_quadrant - 0.5)) + 0.1 * abs(layer_height - 0.5)
            bonuses.append(0.3 * diversity_score)
        return bonuses

    def _residual_coverage_distribution(self) -> np.ndarray:
        free_mask = self.world.grid == int(VoxelState.FREE)
        remaining_mask = free_mask & (self.global_coverage.base_states != int(VoxelState.COVERED))
        quadrant_features: list[float] = []
        x_slices = self._split_axis(self.world.grid_size[0], 2)
        y_slices = self._split_axis(self.world.grid_size[1], 2)
        for x_slice in x_slices:
            for y_slice in y_slices:
                region = np.s_[x_slice, y_slice, :]
                quadrant_features.append(self._remaining_ratio(remaining_mask[region], free_mask[region]))
        layer_features = [
            self._remaining_ratio(remaining_mask[:, :, layer], free_mask[:, :, layer])
            for layer in range(self.world.grid_size[2])
        ]
        return np.asarray([*quadrant_features, *layer_features], dtype=np.float32)

    @staticmethod
    def _remaining_ratio(remaining_mask: np.ndarray, free_mask: np.ndarray) -> float:
        return float(np.count_nonzero(remaining_mask) / max(np.count_nonzero(free_mask), 1))

    @staticmethod
    def _split_axis(length: int, parts: int) -> list[slice]:
        boundaries = np.linspace(0, length, parts + 1, dtype=int)
        return [slice(int(boundaries[index]), int(boundaries[index + 1])) for index in range(parts)]

    def _resolve_step_conflicts(self, results: list[ShieldResult]) -> list[ShieldResult]:
        resolved = list(results)
        planned_next_positions = [
            np.asarray(result.path[1], dtype=float)
            if agent.active and len(result.path) > 1
            else np.asarray(agent.current_voxel, dtype=float)
            for agent, result in zip(self.agents, results, strict=True)
        ]
        current_positions = [np.asarray(agent.current_voxel, dtype=float) for agent in self.agents]
        while True:
            conflicted_pairs: list[tuple[int, int]] = []
            for source in range(len(self.agents)):
                for target in range(source + 1, len(self.agents)):
                    next_too_close = (
                        euclidean_distance(planned_next_positions[source], planned_next_positions[target])
                        < self.env_cfg["safe_agent_dist"]
                    )
                    swapping_positions = np.array_equal(
                        planned_next_positions[source],
                        current_positions[target],
                    ) and np.array_equal(
                        planned_next_positions[target],
                        current_positions[source],
                    )
                    if next_too_close or swapping_positions:
                        conflicted_pairs.append((source, target))
            if not conflicted_pairs:
                break
            changed = False
            for source, target in conflicted_pairs:
                source_moving = not np.array_equal(planned_next_positions[source], current_positions[source])
                target_moving = not np.array_equal(planned_next_positions[target], current_positions[target])
                if source_moving and target_moving:
                    loser = max(source, target)
                elif source_moving:
                    loser = source
                elif target_moving:
                    loser = target
                else:
                    continue
                agent = self.agents[loser]
                resolved[loser] = ShieldResult(
                    agent.current_voxel,
                    [agent.current_voxel],
                    results[loser].chosen_index,
                    "hover",
                    "neighbor_conflict",
                )
                planned_next_positions[loser] = current_positions[loser]
                changed = True
            if not changed:
                break
        return resolved

    def _detect_collisions(
        self,
        sampled_segments: list[np.ndarray] | None = None,
        proposed_positions: list[np.ndarray] | None = None,
    ) -> tuple[list[bool], list[bool], list[bool]]:
        collision_flags = [False for _ in self.agents]
        obstacle_collision_flags = [False for _ in self.agents]
        inter_uav_collision_flags = [False for _ in self.agents]
        candidate_positions = (
            proposed_positions
            if proposed_positions is not None
            else [agent.position for agent in self.agents]
        )

        for index, agent in enumerate(self.agents):
            segment = sampled_segments[index] if sampled_segments is not None else np.asarray([agent.position])
            obstacle_hit = any(
                self.world.is_obstacle(tuple(int(round(value)) for value in sample))
                for sample in segment
            )
            if obstacle_hit:
                collision_flags[index] = True
                obstacle_collision_flags[index] = True

        for source in range(len(self.agents)):
            for target in range(source + 1, len(self.agents)):
                if euclidean_distance(candidate_positions[source], candidate_positions[target]) < self.env_cfg[
                    "safe_agent_dist"
                ]:
                    collision_flags[source] = True
                    collision_flags[target] = True
                    inter_uav_collision_flags[source] = True
                    inter_uav_collision_flags[target] = True
        return collision_flags, obstacle_collision_flags, inter_uav_collision_flags

    def _apply_motion_with_soft_collision_recovery(
        self,
        previous_positions: list[np.ndarray],
        proposed_positions: list[np.ndarray],
        proposed_velocities: list[np.ndarray],
        collision_flags: list[bool],
    ) -> None:
        for index, agent in enumerate(self.agents):
            if collision_flags[index]:
                agent.update_motion(previous_positions[index], np.zeros(3, dtype=float))
                continue
            agent.update_motion(proposed_positions[index], proposed_velocities[index])

    def _build_info(
        self,
        new_coverage: list[int],
        repeated_coverage: list[int],
        unknown_reduction: list[int],
        collision_flags: list[bool],
        obstacle_collision_flags: list[bool],
        inter_uav_collision_flags: list[bool],
        path_lengths: list[float],
        mean_accelerations: list[float],
        max_accelerations: list[float],
        smoothness_costs: list[float],
        team_new_coverage: int,
        terminated: bool,
        truncated: bool,
        planner_statuses: list[str],
        shield_statuses: list[str],
        hover_reasons: list[str],
        goal_conflict_resolutions: int,
        late_reassignment_applied: bool,
        active_flags: list[bool],
    ) -> dict[str, Any]:
        total_observations = sum(new_coverage) + sum(repeated_coverage)
        coverage_complete = self.global_coverage.coverage_ratio(self.world) >= self.env_cfg[
            "target_coverage_ratio"
        ]
        all_failed = all(not agent.active for agent in self.agents)
        if coverage_complete:
            termination_reason = "coverage_target"
        elif all_failed:
            termination_reason = "all_failed"
        elif truncated:
            termination_reason = "max_steps"
        else:
            termination_reason = "running"
        return {
            "coverage_ratio": self.global_coverage.coverage_ratio(self.world),
            "new_coverage": new_coverage,
            "repeated_coverage": repeated_coverage,
            "repeated_coverage_ratio": repeated_coverage_ratio(sum(repeated_coverage), total_observations),
            "unknown_reduction": unknown_reduction,
            "collision_count": int(sum(collision_flags)),
            "obstacle_collision_count": int(sum(obstacle_collision_flags)),
            "inter_uav_collision_count": int(sum(inter_uav_collision_flags)),
            "average_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
            "trajectory_lengths": path_lengths,
            "mean_acceleration": float(np.mean(mean_accelerations)) if mean_accelerations else 0.0,
            "max_acceleration": float(np.max(max_accelerations)) if max_accelerations else 0.0,
            "smoothness_cost": float(np.mean(smoothness_costs)) if smoothness_costs else 0.0,
            "planner_type": self.motion_planner_type,
            "episode_length": self.step_count,
            "communication_links": int(np.count_nonzero(self.adjacency) // 2),
            "physical_communication_links": int(np.count_nonzero(self.physical_adjacency) // 2),
            "effective_communication_links": int(np.count_nonzero(self.adjacency) // 2),
            "global_sync_applied": bool(self.global_sync_applied),
            "team_new_coverage": team_new_coverage,
            "zero_gain_streak": int(self.zero_gain_streak),
            "planner_statuses": planner_statuses,
            "planner_failure_count": sum(status.startswith("failed") for status in planner_statuses),
            "shield_statuses": shield_statuses,
            "hover_reasons": hover_reasons,
            "goal_conflict_resolutions": int(goal_conflict_resolutions),
            "late_reassignment_applied": bool(late_reassignment_applied),
            "active_flags": active_flags,
            "terminated": terminated,
            "truncated": truncated,
            "success": bool(coverage_complete),
            "termination_reason": termination_reason,
            "frontier_counts": [len(frontiers) for frontiers in self.frontier_sets],
        }
