from safe_ctde_mace.marl.trainer import EpisodeTrace
from safe_ctde_mace.utils.reporting import summarize_trace_diagnostics


def test_trace_diagnostics_count_hover_reasons() -> None:
    trace = EpisodeTrace(
        coverage_curve=[0.1, 0.2, 0.2],
        team_new_coverage=[0, 3, 0],
        repeated_coverage_ratio=[0.0, 0.2, 1.0],
        communication_links=[0, 1, 1],
        physical_communication_links=[0, 1, 1],
        effective_communication_links=[0, 1, 1],
        global_sync_applied=[False, False, False],
        collision_count=[0, 0, 0],
        frontier_counts=[4, 5, 5],
        hover_counts=[0, 2, 2],
        hover_reasons=[
            ["", ""],
            ["neighbor_conflict", "planner_unavailable"],
            ["no_valid_candidate", "shield_rejected"],
        ],
        goal_conflict_resolutions=[0, 1, 2],
        late_reassignment_applied=[False, True, True],
        adjusted_counts=[0, 0, 0],
        planner_failure_counts=[0, 0, 1],
        zero_gain_streaks=[0, 0, 1],
        planner_statuses=[["reset", "reset"], ["optimized", "failed"], ["optimized", "optimized"]],
        shield_statuses=[["reset", "reset"], ["hover", "hover"], ["hover", "hover"]],
        active_flags=[[True, True], [True, True], [True, True]],
    )

    diagnostics = summarize_trace_diagnostics(trace)

    assert diagnostics["neighbor_conflict_hovers"] == 1
    assert diagnostics["planner_unavailable_hovers"] == 1
    assert diagnostics["no_valid_candidate_hovers"] == 1
    assert diagnostics["other_hover_total"] == 1
    assert diagnostics["goal_conflict_resolution_total"] == 3
    assert diagnostics["late_reassignment_steps"] == 2
