from copy import deepcopy

import numpy as np

from safe_ctde_mace.envs.multi_uav_env import MultiUAVCoverageEnv
from safe_ctde_mace.mapping.coverage_map import CoverageMap, SensorUpdate
from safe_ctde_mace.mapping.frontier_detector import CandidateSet, FrontierDetector
from safe_ctde_mace.mapping.voxel_map import VoxelState
from safe_ctde_mace.planning.ego_planner import PlannerResult
from safe_ctde_mace.utils.config import load_config


def _small_config() -> dict:
    config = deepcopy(load_config())
    config["environment"]["grid_size"] = [8, 8, 4]
    config["environment"]["sensor_range"] = 1.8
    config["environment"]["comm_range"] = 10.0
    config["environment"]["safe_agent_dist"] = 1.0
    config["environment"]["local_patch_radius"] = 2
    config["environment"]["num_frontier_candidates"] = 4
    config["environment"]["max_steps"] = 5
    config["environment"]["target_coverage_ratio"] = 0.99
    config["environment"]["initial_positions"] = [[1, 1, 1], [1, 6, 1]]
    config["environment"]["num_uavs"] = 2
    config["environment"]["max_neighbors"] = 1
    config["environment"]["obstacle_generation"] = {
        "random_boxes": 0,
        "min_box_size": [1, 1, 1],
        "max_box_size": [1, 1, 1],
        "manual_boxes": [],
    }
    return config


def test_env_reset_builds_expected_observation_shapes() -> None:
    env = MultiUAVCoverageEnv(_small_config())
    observations, info = env.reset(seed=11)

    assert len(observations) == 2
    first = observations[0]
    assert first["local_voxel_map"].shape == (7, 5, 5, 5)
    assert first["self_state"].shape == (9,)
    assert first["neighbor_states"].shape == (1, 9)
    assert first["candidate_features"].shape == (4, 10)
    assert first["action_mask"].shape == (4,)
    assert info["coverage_ratio"] > 0.0


def test_env_step_runs_with_masked_fallback_action() -> None:
    env = MultiUAVCoverageEnv(_small_config())
    observations, _ = env.reset(seed=11)
    actions = [999, 999]
    next_obs, rewards, terminated, truncated, info = env.step(actions)

    assert len(next_obs) == len(observations)
    assert len(rewards) == 2
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "shield_statuses" in info
    assert info["team_new_coverage"] >= 0


def test_env_truncates_after_max_steps() -> None:
    config = _small_config()
    config["environment"]["max_steps"] = 1
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)
    _, _, _, truncated, info = env.step([0, 0])

    assert truncated is True
    assert info["episode_length"] == 1


def test_env_observation_action_masks_are_boolean() -> None:
    env = MultiUAVCoverageEnv(_small_config())
    observations, _ = env.reset(seed=11)
    assert all(obs["action_mask"].dtype == np.bool_ for obs in observations)


def test_env_deconflicts_neighbor_goals_before_reservation_broadcast() -> None:
    env = MultiUAVCoverageEnv(_small_config())
    env.reset(seed=11)
    env.adjacency = np.asarray([[False, True], [True, False]], dtype=bool)
    env.current_candidates = [
        CandidateSet(
            goals=np.asarray([[2, 2, 1], [5, 5, 1]], dtype=np.int16),
            features=np.zeros((2, 7), dtype=np.float32),
            action_mask=np.asarray([True, True]),
            frontier_voxels=set(),
        ),
        CandidateSet(
            goals=np.asarray([[2, 2, 1], [6, 5, 1]], dtype=np.int16),
            features=np.zeros((2, 7), dtype=np.float32),
            action_mask=np.asarray([True, True]),
            frontier_voxels=set(),
        ),
    ]

    actions, resolutions = env._deconflict_selected_actions([0, 0])

    assert actions == [0, 1]
    assert resolutions == 1


def test_normalized_reward_decreases_with_repeat_ratio() -> None:
    env = MultiUAVCoverageEnv(_small_config())
    env.reset(seed=11)
    observed = {(index, 0, 0) for index in range(10)}
    local_updates = [SensorUpdate(0, 0, 5, observed), SensorUpdate(0, 0, 5, observed)]
    low_repeat = [SensorUpdate(5, 1, 0, observed), SensorUpdate(5, 1, 0, observed)]
    high_repeat = [SensorUpdate(5, 9, 0, observed), SensorUpdate(5, 9, 0, observed)]

    low_rewards = env.compute_reward(
        local_updates=local_updates,
        global_updates=low_repeat,
        collision_flags=[False, False],
        obstacle_collision_flags=[False, False],
        inter_uav_collision_flags=[False, False],
        step_distances=[0.0, 0.0],
        reserved_penalties=[0.0, 0.0],
        team_new_coverage=10,
    )
    high_rewards = env.compute_reward(
        local_updates=local_updates,
        global_updates=high_repeat,
        collision_flags=[False, False],
        obstacle_collision_flags=[False, False],
        inter_uav_collision_flags=[False, False],
        step_distances=[0.0, 0.0],
        reserved_penalties=[0.0, 0.0],
        team_new_coverage=10,
    )

    assert all(low > high for low, high in zip(low_rewards, high_rewards, strict=True))


def test_normalized_reward_keeps_repeat_penalty_bounded() -> None:
    env = MultiUAVCoverageEnv(_small_config())
    env.reset(seed=11)
    observed = {(index, 0, 0) for index in range(1000)}
    rewards = env.compute_reward(
        local_updates=[SensorUpdate(0, 0, 0, observed), SensorUpdate(0, 0, 0, observed)],
        global_updates=[SensorUpdate(0, 1000, 0, observed), SensorUpdate(0, 1000, 0, observed)],
        collision_flags=[False, False],
        obstacle_collision_flags=[False, False],
        inter_uav_collision_flags=[False, False],
        step_distances=[0.0, 0.0],
        reserved_penalties=[0.0, 0.0],
        team_new_coverage=0,
    )

    assert all(-5.0 < reward < 1.0 for reward in rewards)


def test_env_resolves_next_step_inter_uav_conflicts() -> None:
    config = _small_config()
    config["environment"]["initial_positions"] = [[1, 1, 1], [3, 1, 1]]
    config["environment"]["safe_agent_dist"] = 1.0
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)
    results = [
        env.safety_shield.select_safe_goal(
            current_position=(1, 1, 1),
            candidate_goals=np.asarray([[2, 1, 1]], dtype=np.int16),
            action_mask=np.asarray([True]),
            chosen_action=0,
            knowledge_states=np.full((8, 8, 4), 1, dtype=np.int8),
            neighbor_states=[],
            planner=env.planner,
        ),
        env.safety_shield.select_safe_goal(
            current_position=(3, 1, 1),
            candidate_goals=np.asarray([[2, 1, 1]], dtype=np.int16),
            action_mask=np.asarray([True]),
            chosen_action=0,
            knowledge_states=np.full((8, 8, 4), 1, dtype=np.int8),
            neighbor_states=[],
            planner=env.planner,
        ),
    ]

    resolved = env._resolve_step_conflicts(results)
    assert resolved[0].status == "safe"
    assert resolved[1].status == "hover"
    assert resolved[1].hover_reason == "neighbor_conflict"


def test_env_keeps_both_agents_hovering_when_no_single_yield_is_safe() -> None:
    config = _small_config()
    config["environment"]["initial_positions"] = [[1, 1, 1], [3, 1, 1]]
    config["environment"]["safe_agent_dist"] = 1.5
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)
    results = [
        env.safety_shield.select_safe_goal(
            current_position=(1, 1, 1),
            candidate_goals=np.asarray([[2, 1, 1]], dtype=np.int16),
            action_mask=np.asarray([True]),
            chosen_action=0,
            knowledge_states=np.full((8, 8, 4), 1, dtype=np.int8),
            neighbor_states=[],
            planner=env.planner,
        ),
        env.safety_shield.select_safe_goal(
            current_position=(3, 1, 1),
            candidate_goals=np.asarray([[2, 1, 1]], dtype=np.int16),
            action_mask=np.asarray([True]),
            chosen_action=0,
            knowledge_states=np.full((8, 8, 4), 1, dtype=np.int8),
            neighbor_states=[],
            planner=env.planner,
        ),
    ]

    resolved = env._resolve_step_conflicts(results)

    assert [item.status for item in resolved] == ["hover", "hover"]
    assert [item.hover_reason for item in resolved] == ["neighbor_conflict", "neighbor_conflict"]


def test_env_reports_continuous_planner_metrics() -> None:
    config = _small_config()
    config["environment"]["planner_type"] = "ego"
    config["environment"]["trajectory_execution_dt"] = 0.5
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)
    _, _, _, _, info = env.step([0, 0])
    assert info["planner_type"] == "ego"
    assert "trajectory_lengths" in info
    assert "mean_acceleration" in info
    assert "planner_statuses" in info


def test_env_reports_ego_planner_failures() -> None:
    config = _small_config()
    config["environment"]["planner_type"] = "ego"
    config["environment"]["num_uavs"] = 1
    config["environment"]["initial_positions"] = [[1, 1, 1]]
    config["environment"]["max_neighbors"] = 0
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)
    env.ego_planner.plan_with_status = lambda *_args, **_kwargs: PlannerResult(  # type: ignore[method-assign]
        None,
        "failed_all_fallbacks",
    )
    _, _, _, _, info = env.step([0])

    assert info["planner_statuses"] == ["failed_all_fallbacks"]
    assert info["planner_failure_count"] == 1
    assert info["shield_statuses"] == ["hover"]
    assert info["hover_reasons"] == ["planner_unavailable"]


def test_large_qmix_env_builds_three_uav_observations() -> None:
    env = MultiUAVCoverageEnv(load_config("safe_ctde_mace/configs/qmix_ego_large.yaml"))
    observations, info = env.reset(seed=7)

    assert len(observations) == 3
    first = observations[0]
    assert first["local_voxel_map"].shape == (7, 9, 9, 9)
    assert first["neighbor_states"].shape == (2, 9)
    assert first["candidate_features"].shape == (6, 11)
    assert info["global_sync_applied"] is False
    assert info["effective_communication_links"] == info["physical_communication_links"]


def test_global_state_appends_residual_coverage_distribution() -> None:
    env = MultiUAVCoverageEnv(_small_config())
    env.reset(seed=11)
    env.global_coverage.base_states.fill(int(VoxelState.UNKNOWN))
    env.global_coverage.mark_covered(
        [
            *[(x, y, z) for x in range(0, 4) for y in range(0, 4) for z in range(4)],
            *[(x, y, 0) for x in range(4, 8) for y in range(4, 8)],
        ]
    )

    state = env.get_global_state()
    residual_features = state[5:13]

    assert state.shape[0] == 39
    np.testing.assert_allclose(
        residual_features,
        np.asarray([0.0, 1.0, 1.0, 0.75, 0.5, 0.75, 0.75, 0.75], dtype=np.float32),
    )


def test_late_reassignment_triggers_only_after_late_stall() -> None:
    config = _small_config()
    config["environment"]["late_reassign_enabled"] = True
    config["environment"]["late_reassign_min_coverage"] = 0.70
    config["environment"]["late_reassign_zero_gain_streak"] = 5
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)
    env.zero_gain_streak = 5
    assert env._should_apply_late_reassignment() is False

    covered = [(x, y, z) for x in range(8) for y in range(8) for z in range(3)]
    env.global_coverage.mark_covered(covered)

    assert env._should_apply_late_reassignment() is True


def test_late_reassignment_triggers_on_low_recent_gain_window() -> None:
    config = _small_config()
    config["environment"]["late_reassign_enabled"] = True
    config["environment"]["late_reassign_min_coverage"] = 0.50
    config["environment"]["late_reassign_zero_gain_streak"] = 99
    config["environment"]["late_reassign_window"] = 3
    config["environment"]["late_reassign_max_mean_gain"] = 2.0
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)
    env.global_coverage.mark_covered([(x, y, z) for x in range(8) for y in range(8) for z in range(3)])
    env.recent_team_gains = [3, 1, 2]

    assert env._should_apply_late_reassignment() is True


def test_late_reassignment_prefers_unique_nearest_goals() -> None:
    env = MultiUAVCoverageEnv(_small_config())
    env.reset(seed=11)
    layout = env.frontier_detector.feature_layout
    first_features = np.zeros((2, layout.size), dtype=np.float32)
    first_features[:, 5] = np.asarray([1.0, 3.0], dtype=np.float32)
    second_features = np.zeros((2, layout.size), dtype=np.float32)
    second_features[:, 5] = np.asarray([2.0, 4.0], dtype=np.float32)
    env.current_candidates = [
        CandidateSet(
            goals=np.asarray([[2, 2, 1], [4, 4, 1]], dtype=np.int16),
            features=first_features,
            action_mask=np.asarray([True, True]),
            frontier_voxels=set(),
        ),
        CandidateSet(
            goals=np.asarray([[2, 2, 1], [6, 6, 1]], dtype=np.int16),
            features=second_features,
            action_mask=np.asarray([True, True]),
            frontier_voxels=set(),
        ),
    ]

    assert env._late_reassign_actions([0, 0]) == [0, 1]


def test_late_reassignment_uses_uncovered_density_as_distance_tiebreaker() -> None:
    env = MultiUAVCoverageEnv(_small_config())
    env.reset(seed=11)
    layout = env.frontier_detector.feature_layout
    features = np.zeros((2, layout.size), dtype=np.float32)
    features[0, 5] = 1.0
    features[0, layout.uncovered_density] = 0.1
    features[1, 5] = 1.0
    features[1, layout.uncovered_density] = 0.9
    env.current_candidates = [
        CandidateSet(
            goals=np.asarray([[2, 2, 1], [4, 4, 1]], dtype=np.int16),
            features=features.copy(),
            action_mask=np.asarray([True, True]),
            frontier_voxels=set(),
        ),
        CandidateSet(
            goals=np.asarray([[6, 6, 1], [7, 7, 1]], dtype=np.int16),
            features=features.copy(),
            action_mask=np.asarray([True, True]),
            frontier_voxels=set(),
        ),
    ]

    assert env._late_reassign_actions([0, 0]) == [1, 1]


def test_exploration_bonus_reads_dynamic_tail_features_for_three_uav_layout() -> None:
    config = _small_config()
    config["environment"]["num_uavs"] = 3
    config["environment"]["initial_positions"] = [[1, 1, 1], [1, 6, 1], [6, 1, 1]]
    config["environment"]["max_neighbors"] = 2
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)
    env.zero_gain_streak = 3
    covered = [(x, y, z) for x in range(4) for y in range(8) for z in range(4)]
    env.global_coverage.mark_covered(covered)

    layout = env.frontier_detector.feature_layout
    base_features = np.zeros((2, layout.size), dtype=np.float32)
    base_features[0, layout.uncovered_density] = 1.0
    base_features[0, layout.grid_quadrant] = 0.5
    base_features[0, layout.layer_height] = 1.0
    base_features[1, layout.uncovered_density] = 0.0
    base_features[1, 9] = 99.0
    env.current_candidates = [
        CandidateSet(
            goals=np.asarray([[2, 2, 1], [3, 3, 1]], dtype=np.int16),
            features=base_features.copy(),
            action_mask=np.asarray([True, True]),
            frontier_voxels=set(),
        )
        for _ in range(3)
    ]

    bonuses = env._compute_exploration_bonus()

    np.testing.assert_allclose(bonuses, np.asarray([0.195, 0.195, 0.195], dtype=float))


def test_exploration_bonus_uses_selected_action_features() -> None:
    config = _small_config()
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)
    env.zero_gain_streak = 3
    env.global_coverage.mark_covered([(x, y, z) for x in range(4) for y in range(8) for z in range(4)])
    layout = env.frontier_detector.feature_layout
    features = np.zeros((2, layout.size), dtype=np.float32)
    features[0, layout.uncovered_density] = 1.0
    features[1, layout.uncovered_density] = 0.0
    env.current_candidates = [
        CandidateSet(
            goals=np.asarray([[2, 2, 1], [3, 3, 1]], dtype=np.int16),
            features=features.copy(),
            action_mask=np.asarray([True, True]),
            frontier_voxels=set(),
        )
        for _ in range(2)
    ]

    bonuses = env._compute_exploration_bonus([1, 1])

    np.testing.assert_allclose(bonuses, np.asarray([0.03, 0.03], dtype=float))


def test_global_sync_uses_effective_links_without_changing_physical_links() -> None:
    config = _small_config()
    config["environment"]["comm_range"] = 1.0
    config["environment"]["global_sync_interval"] = 1
    env = MultiUAVCoverageEnv(config)
    observations, info = env.reset(seed=11)

    assert info["physical_communication_links"] == 0
    assert info["effective_communication_links"] == 1
    assert info["communication_links"] == 1
    assert info["global_sync_applied"] is True
    assert observations[0]["neighbor_states"][0].any()


def test_inter_uav_collision_is_soft_and_recovers_previous_positions() -> None:
    config = _small_config()
    config["environment"]["initial_positions"] = [[1, 1, 1], [3, 1, 1]]
    config["environment"]["safe_agent_dist"] = 1.5
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)

    previous_positions = [agent.position.copy() for agent in env.agents]
    collision_flags, obstacle_flags, inter_flags = env._detect_collisions(
        sampled_segments=[
            np.asarray([[1, 1, 1], [2, 1, 1]], dtype=float),
            np.asarray([[3, 1, 1], [2, 1, 1]], dtype=float),
        ],
        proposed_positions=[
            np.asarray([2, 1, 1], dtype=float),
            np.asarray([2, 1, 1], dtype=float),
        ],
    )
    env._apply_motion_with_soft_collision_recovery(
        previous_positions=previous_positions,
        proposed_positions=[
            np.asarray([2, 1, 1], dtype=float),
            np.asarray([2, 1, 1], dtype=float),
        ],
        proposed_velocities=[
            np.asarray([1, 0, 0], dtype=float),
            np.asarray([-1, 0, 0], dtype=float),
        ],
        collision_flags=collision_flags,
    )

    assert collision_flags == [True, True]
    assert obstacle_flags == [False, False]
    assert inter_flags == [True, True]
    assert all(agent.active for agent in env.agents)
    np.testing.assert_array_equal(env.agents[0].position, previous_positions[0])
    np.testing.assert_array_equal(env.agents[1].position, previous_positions[1])
    assert all(np.array_equal(agent.velocity, np.zeros(3)) for agent in env.agents)


def test_obstacle_collision_is_soft_and_recovers_previous_position() -> None:
    config = _small_config()
    config["environment"]["num_uavs"] = 1
    config["environment"]["initial_positions"] = [[1, 1, 1]]
    config["environment"]["max_neighbors"] = 0
    config["environment"]["obstacle_generation"]["manual_boxes"] = [
        {"min_corner": [2, 1, 1], "max_corner": [2, 1, 1]}
    ]
    env = MultiUAVCoverageEnv(config)
    env.reset(seed=11)

    previous_positions = [env.agents[0].position.copy()]
    collision_flags, obstacle_flags, inter_flags = env._detect_collisions(
        sampled_segments=[np.asarray([[1, 1, 1], [2, 1, 1]], dtype=float)],
        proposed_positions=[np.asarray([2, 1, 1], dtype=float)],
    )
    env._apply_motion_with_soft_collision_recovery(
        previous_positions=previous_positions,
        proposed_positions=[np.asarray([2, 1, 1], dtype=float)],
        proposed_velocities=[np.asarray([1, 0, 0], dtype=float)],
        collision_flags=collision_flags,
    )

    assert collision_flags == [True]
    assert obstacle_flags == [True]
    assert inter_flags == [False]
    assert env.agents[0].active is True
    np.testing.assert_array_equal(env.agents[0].position, previous_positions[0])
    np.testing.assert_array_equal(env.agents[0].velocity, np.zeros(3))


def test_coverage_patch_matches_reference_sampling() -> None:
    coverage = CoverageMap((5, 4, 3))
    coverage.mark_free([(1, 1, 1), (4, 3, 2)])
    coverage.mark_covered([(2, 2, 1)])
    coverage.mark_obstacle([(0, 0, 0)])
    states = coverage.as_array()
    patch = coverage.get_patch(center=(0, 1, 1), radius=2, states=states)

    expected = np.full((5, 5, 5), int(VoxelState.UNKNOWN), dtype=np.int8)
    for local_index in np.ndindex(expected.shape):
        world_index = tuple(np.asarray((0, 1, 1)) + np.asarray(local_index) - 2)
        if all(0 <= value < limit for value, limit in zip(world_index, states.shape, strict=True)):
            expected[local_index] = states[world_index]

    np.testing.assert_array_equal(patch, expected)


def test_frontier_detection_matches_reference_definition() -> None:
    states = np.full((4, 4, 3), int(VoxelState.UNKNOWN), dtype=np.int8)
    states[1:3, 1:3, 1] = int(VoxelState.FREE)
    states[2, 2, 1] = int(VoxelState.COVERED)
    detector = FrontierDetector(num_candidates=4, sensor_range=1.0)

    expected: set[tuple[int, int, int]] = set()
    for index in np.argwhere(np.isin(states, [int(VoxelState.FREE), int(VoxelState.COVERED)])):
        voxel = tuple(int(value) for value in index)
        neighbors = [
            (voxel[0] + 1, voxel[1], voxel[2]),
            (voxel[0] - 1, voxel[1], voxel[2]),
            (voxel[0], voxel[1] + 1, voxel[2]),
            (voxel[0], voxel[1] - 1, voxel[2]),
            (voxel[0], voxel[1], voxel[2] + 1),
            (voxel[0], voxel[1], voxel[2] - 1),
        ]
        if any(
            0 <= x < states.shape[0]
            and 0 <= y < states.shape[1]
            and 0 <= z < states.shape[2]
            and states[x, y, z] == int(VoxelState.UNKNOWN)
            for x, y, z in neighbors
        ):
            expected.add(voxel)

    assert detector.detect_frontiers(states) == expected
