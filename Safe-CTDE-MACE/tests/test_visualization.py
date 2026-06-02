from copy import deepcopy

import matplotlib.pyplot as plt

from safe_ctde_mace.envs.multi_uav_env import MultiUAVCoverageEnv
from safe_ctde_mace.utils.config import load_config
from safe_ctde_mace.marl.trainer import EpisodeSummary, EpisodeTrace
from safe_ctde_mace.utils.replay import capture_episode_frame, start_episode_replay
from safe_ctde_mace.utils.visualization import (
    plot_coverage_curve,
    plot_episode,
    plot_episode_animation,
    plot_episode_diagnostics,
    plot_evaluation_summary,
    plot_planner_comparison,
    plot_training_history,
)


def _config() -> dict:
    config = deepcopy(load_config())
    config["environment"]["grid_size"] = [6, 6, 3]
    config["environment"]["num_uavs"] = 1
    config["environment"]["initial_positions"] = [[1, 1, 1]]
    config["environment"]["max_neighbors"] = 0
    config["environment"]["num_frontier_candidates"] = 2
    config["environment"]["local_patch_radius"] = 1
    config["environment"]["obstacle_generation"] = {
        "random_boxes": 0,
        "min_box_size": [1, 1, 1],
        "max_box_size": [1, 1, 1],
        "manual_boxes": [],
    }
    return config


def test_visualization_helpers_return_figures() -> None:
    env = MultiUAVCoverageEnv(_config())
    env.reset()
    episode_figure = plot_episode(env)
    curve_figure = plot_coverage_curve([0.1, 0.2, 0.3])
    history = [
        EpisodeSummary(1.0, 0.5, 10, False, 0.2, 0, 0.1, False, "max_steps"),
        EpisodeSummary(2.0, 0.9, 8, True, 0.1, 0, 0.05, False, "coverage_target"),
    ]
    trace = EpisodeTrace(
        coverage_curve=[0.1, 0.3, 0.9],
        team_new_coverage=[0, 5, 8],
        repeated_coverage_ratio=[0.0, 0.1, 0.2],
        communication_links=[1, 1, 1],
        physical_communication_links=[1, 1, 1],
        effective_communication_links=[1, 1, 1],
        global_sync_applied=[False, False, False],
        collision_count=[0, 0, 0],
        frontier_counts=[4, 5, 2],
        hover_counts=[0, 1, 0],
        adjusted_counts=[0, 0, 0],
        planner_failure_counts=[0, 1, 0],
        zero_gain_streaks=[0, 1, 0],
        planner_statuses=[["planned"], ["failed"], ["planned"]],
        shield_statuses=[["safe"], ["hover"], ["safe"]],
        active_flags=[[True], [True], [True]],
    )
    train_figure = plot_training_history(history)
    diagnostics_figure = plot_episode_diagnostics(trace)
    eval_figure = plot_evaluation_summary(history)
    planner_figure = plot_planner_comparison(
        [
            {
                "planner_type": "astar",
                "average_path_length": 2.0,
                "mean_acceleration": 1.0,
                "smoothness_cost": 1.5,
            },
            {
                "planner_type": "ego",
                "average_path_length": 2.1,
                "mean_acceleration": 0.4,
                "smoothness_cost": 0.3,
            },
        ]
    )
    assert episode_figure is not None
    assert episode_figure.axes[0].lines
    assert curve_figure is not None
    assert train_figure is not None
    assert diagnostics_figure is not None
    assert eval_figure is not None
    assert planner_figure is not None
    plt.close(episode_figure)
    plt.close(curve_figure)
    plt.close(train_figure)
    plt.close(diagnostics_figure)
    plt.close(eval_figure)
    plt.close(planner_figure)


def test_episode_animation_writes_gif(tmp_path) -> None:
    env = MultiUAVCoverageEnv(_config())
    observations, _ = env.reset()
    replay = start_episode_replay(env)
    actions = [int(next(iter(obs["action_mask"].nonzero()[0]), 0)) for obs in observations]
    env.step(actions)
    replay.frames.append(capture_episode_frame(env))

    animation = plot_episode_animation(replay, tmp_path / "episode.gif", interval_ms=1)

    assert (tmp_path / "episode.gif").exists()
    assert (tmp_path / "episode.gif").stat().st_size > 0
