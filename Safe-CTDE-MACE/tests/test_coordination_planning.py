import numpy as np

from safe_ctde_mace.communication.comm_graph import CommGraph
from safe_ctde_mace.communication.map_fusion import MapFusion
from safe_ctde_mace.mapping.coverage_map import CoverageMap
from safe_ctde_mace.mapping.frontier_detector import CandidateFeatureLayout, FrontierDetector, candidate_score
from safe_ctde_mace.mapping.voxel_map import VoxelState
from safe_ctde_mace.planning.astar_3d import AStar3D
from safe_ctde_mace.planning.ego_planner import EGOStylePlanner
from safe_ctde_mace.planning.safety_shield import SafetyShield
from safe_ctde_mace.utils.metrics import trajectory_metrics_from_points


def test_comm_graph_neighbors() -> None:
    graph = CommGraph(comm_range=3.0)
    positions = [(0, 0, 0), (2, 0, 0), (5, 0, 0)]
    adjacency = graph.adjacency_matrix(positions)
    assert adjacency.tolist() == [
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ]


def test_map_fusion_priority_and_compression() -> None:
    first = CoverageMap((3, 3, 3))
    second = CoverageMap((3, 3, 3))
    first.mark_free([(0, 0, 0)])
    first.mark_obstacle([(2, 2, 2)])
    first.reserve([(1, 1, 1)])
    second.mark_covered([(0, 0, 0)])
    second.mark_covered([(2, 2, 2)])
    second.mark_obstacle([(1, 1, 1)])
    MapFusion.fuse_neighbors([first, second], [[1], [0]])

    assert first.base_states[0, 0, 0] == int(VoxelState.COVERED)
    assert first.base_states[1, 1, 1] == int(VoxelState.OBSTACLE)
    assert first.base_states[2, 2, 2] == int(VoxelState.OBSTACLE)
    flat = MapFusion.compress_indices([(1, 1, 1), (2, 2, 2)], first.shape)
    assert MapFusion.decompress_indices(flat, first.shape) == [(1, 1, 1), (2, 2, 2)]


def test_frontier_detector_padding_and_mask() -> None:
    states = np.full((5, 5, 3), int(VoxelState.UNKNOWN), dtype=np.int8)
    states[2, 2, 1] = int(VoxelState.COVERED)
    detector = FrontierDetector(num_candidates=4, sensor_range=2.0)
    candidates = detector.generate_candidates(states, (2, 2, 1))

    assert len(candidates.frontier_voxels) == 1
    assert candidates.action_mask.tolist() == [True, False, False, False]
    assert candidates.goals[0].tolist() == [2, 2, 1]


def test_frontier_detector_filters_unreachable_frontiers() -> None:
    states = np.full((5, 5, 1), int(VoxelState.UNKNOWN), dtype=np.int8)
    states[0, 0, 0] = int(VoxelState.COVERED)
    states[4, 4, 0] = int(VoxelState.COVERED)
    detector = FrontierDetector(num_candidates=3, sensor_range=1.0)
    candidates = detector.generate_candidates(states, (0, 0, 0))

    assert candidates.goals[0].tolist() == [0, 0, 0]
    assert [4, 4, 0] not in candidates.goals.tolist()


def test_frontier_detector_normalizes_boundary_information_gain() -> None:
    states = np.full((7, 7, 3), int(VoxelState.UNKNOWN), dtype=np.int8)
    detector = FrontierDetector(num_candidates=3, sensor_range=1.5)

    boundary_gain = detector._expected_information_gain(states, (0, 3, 1))
    interior_gain = detector._expected_information_gain(states, (3, 3, 1))

    assert boundary_gain == interior_gain


def test_frontier_detector_appends_assignment_margins() -> None:
    states = np.full((5, 5, 3), int(VoxelState.FREE), dtype=np.int8)
    detector = FrontierDetector(num_candidates=3, sensor_range=1.5, max_neighbors=2)

    features = detector.compute_features(
        states,
        candidate=(2, 2, 1),
        uav_position=(0, 0, 1),
        neighbor_positions=[(2, 5, 1), (6, 2, 1)],
        path_cost=4.0,
    )

    np.testing.assert_allclose(features[6:8], np.asarray([-1.0, 0.0], dtype=np.float32))


def test_frontier_detector_prefers_spatially_separated_candidates_when_configured() -> None:
    detector = FrontierDetector(
        num_candidates=3,
        sensor_range=1.5,
        candidate_min_separation=3.0,
    )
    scored = [
        (10.0, (0, 0, 0), np.zeros(detector.feature_layout.size, dtype=np.float32)),
        (9.0, (1, 0, 0), np.zeros(detector.feature_layout.size, dtype=np.float32)),
        (8.0, (5, 0, 0), np.zeros(detector.feature_layout.size, dtype=np.float32)),
        (7.0, (9, 0, 0), np.zeros(detector.feature_layout.size, dtype=np.float32)),
    ]

    selected = detector._select_diverse_candidates(scored)

    assert [goal for _, goal, _ in selected] == [(0, 0, 0), (5, 0, 0), (9, 0, 0)]


def test_candidate_feature_layout_offsets_follow_neighbor_count() -> None:
    zero_neighbors = CandidateFeatureLayout(max_neighbors=0)
    one_neighbor = CandidateFeatureLayout(max_neighbors=1)
    two_neighbors = CandidateFeatureLayout(max_neighbors=2)

    assert zero_neighbors.assignment_margin_slice == slice(6, 6)
    assert zero_neighbors.grid_quadrant == 6
    assert zero_neighbors.layer_height == 7
    assert zero_neighbors.uncovered_density == 8
    assert zero_neighbors.size == 9

    assert one_neighbor.assignment_margin_slice == slice(6, 7)
    assert one_neighbor.grid_quadrant == 7
    assert one_neighbor.layer_height == 8
    assert one_neighbor.uncovered_density == 9
    assert one_neighbor.size == 10

    assert two_neighbors.assignment_margin_slice == slice(6, 8)
    assert two_neighbors.grid_quadrant == 8
    assert two_neighbors.layer_height == 9
    assert two_neighbors.uncovered_density == 10
    assert two_neighbors.size == 11


def test_frontier_score_reads_dynamic_tail_features_for_three_uav_layout() -> None:
    detector = FrontierDetector(num_candidates=2, sensor_range=1.5, max_neighbors=2)
    features = np.zeros(detector.feature_layout.size, dtype=np.float32)
    features[detector.feature_layout.uncovered_density] = 1.0
    features[7] = 50.0

    assert detector._score(features) == 0.3


def test_frontier_score_matches_shared_candidate_score_for_real_feature_vector() -> None:
    states = np.full((5, 5, 3), int(VoxelState.UNKNOWN), dtype=np.int8)
    states[2, 2, 1] = int(VoxelState.COVERED)
    detector = FrontierDetector(num_candidates=2, sensor_range=1.5, max_neighbors=2)
    features = detector.compute_features(
        states,
        candidate=(2, 2, 1),
        uav_position=(0, 0, 1),
        neighbor_positions=[(2, 4, 1), (4, 2, 1)],
        path_cost=4.0,
    )

    assert detector._score(features) == candidate_score(features, detector.feature_layout)


def test_astar_reachable_and_unreachable() -> None:
    states = np.full((4, 4, 2), int(VoxelState.FREE), dtype=np.int8)
    planner = AStar3D(connectivity=6)
    path = planner.plan((0, 0, 0), (3, 3, 1), states)
    assert path is not None
    states[1, :, :] = int(VoxelState.OBSTACLE)
    assert planner.plan((0, 0, 0), (3, 3, 1), states) is None


def test_safety_shield_uses_next_feasible_candidate() -> None:
    states = np.full((5, 5, 3), int(VoxelState.FREE), dtype=np.int8)
    states[2, 2, 1] = int(VoxelState.OBSTACLE)
    goals = np.asarray([[2, 2, 1], [4, 4, 1]], dtype=np.int16)
    shield = SafetyShield(safe_obs_dist=1.0, safe_agent_dist=1.0)
    result = shield.select_safe_goal(
        current_position=(0, 0, 1),
        candidate_goals=goals,
        action_mask=np.asarray([True, True]),
        chosen_action=0,
        knowledge_states=states,
        neighbor_states=[],
        planner=AStar3D(),
    )

    assert result.safe_goal == (4, 4, 1)
    assert result.chosen_index == 1
    assert result.status == "safe"


def test_safety_shield_hover_when_no_candidate_is_valid() -> None:
    states = np.full((3, 3, 1), int(VoxelState.UNKNOWN), dtype=np.int8)
    states[1, 1, 0] = int(VoxelState.COVERED)
    shield = SafetyShield(safe_obs_dist=1.0, safe_agent_dist=1.0)
    result = shield.select_safe_goal(
        current_position=(1, 1, 0),
        candidate_goals=np.asarray([[0, 0, 0]], dtype=np.int16),
        action_mask=np.asarray([True]),
        chosen_action=0,
        knowledge_states=states,
        neighbor_states=[],
        planner=AStar3D(),
    )
    assert result.status == "hover"
    assert result.path == [(1, 1, 0)]


def test_ego_style_planner_returns_continuous_trajectory() -> None:
    states = np.full((6, 6, 2), int(VoxelState.FREE), dtype=np.int8)
    planner = EGOStylePlanner(
        max_velocity=1.0,
        max_acceleration=2.0,
        safe_obs_dist=1.0,
        sample_dt=0.25,
    )
    trajectory = planner.plan((0, 0, 0), (5, 5, 1), states)
    assert trajectory is not None
    position, velocity, acceleration = trajectory.sample(min(0.5, trajectory.duration))
    assert position.shape == (3,)
    assert velocity.shape == (3,)
    assert acceleration.shape == (3,)
    assert trajectory.metrics()["path_length"] > 0.0


def test_ego_style_trajectory_is_smoother_than_astar_polyline() -> None:
    states = np.full((6, 6, 1), int(VoxelState.FREE), dtype=np.int8)
    astar_path = AStar3D(connectivity=6).plan((0, 0, 0), (5, 5, 0), states)
    assert astar_path is not None
    astar_metrics = trajectory_metrics_from_points(np.asarray(astar_path, dtype=float))
    ego = EGOStylePlanner(
        max_velocity=1.0,
        max_acceleration=100.0,
        safe_obs_dist=0.0,
        sample_dt=0.25,
        seed_connectivity=6,
    ).plan((0, 0, 0), (5, 5, 0), states)
    assert ego is not None
    assert ego.metrics()["smoothness_cost"] <= astar_metrics["smoothness_cost"]


def test_ego_style_planner_uses_axis_aligned_fallback_after_other_paths_fail() -> None:
    states = np.full((6, 6, 1), int(VoxelState.FREE), dtype=np.int8)
    planner = EGOStylePlanner(
        max_velocity=1.0,
        max_acceleration=2.0,
        safe_obs_dist=1.0,
        sample_dt=0.25,
    )
    validation_calls = 0

    def fake_validation(*_args, **_kwargs) -> bool:
        nonlocal validation_calls
        validation_calls += 1
        return validation_calls == 3

    planner._trajectory_is_valid = fake_validation  # type: ignore[method-assign]
    result = planner.plan_with_status((0, 0, 0), (5, 5, 0), states)

    assert result.trajectory is not None
    assert result.status == "axis_aligned_fallback"
    assert validation_calls == 3
