from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable


def save_episode_summaries(path: str | Path, summaries: Iterable) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(summary) for summary in summaries]
    if not rows:
        return
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_trace(path: str | Path, trace) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(asdict(trace), handle, indent=2)


def summarize_trace_diagnostics(trace) -> dict[str, int | float | None]:
    flattened_hover_reasons = [
        reason
        for step_reasons in trace.hover_reasons
        for reason in step_reasons
        if reason
    ]
    plateau_step = next(
        (
            index
            for index in range(1, len(trace.coverage_curve))
            if all(abs(value - trace.coverage_curve[index]) < 1e-12 for value in trace.coverage_curve[index:])
        ),
        None,
    )
    return {
        "plateau_step": plateau_step,
        "first_hover_step": _first_positive_step(trace.hover_counts),
        "first_collision_step": _first_positive_step(trace.collision_count),
        "first_planner_failure_step": _first_positive_step(trace.planner_failure_counts),
        "first_physical_link_step": _first_positive_step(trace.physical_communication_links),
        "first_effective_link_step": _first_positive_step(trace.effective_communication_links),
        "max_zero_gain_streak": max(trace.zero_gain_streaks, default=0),
        "planner_failure_total": int(sum(trace.planner_failure_counts)),
        "hover_total": int(sum(trace.hover_counts)),
        "no_valid_candidate_hovers": flattened_hover_reasons.count("no_valid_candidate"),
        "neighbor_conflict_hovers": flattened_hover_reasons.count("neighbor_conflict"),
        "planner_unavailable_hovers": flattened_hover_reasons.count("planner_unavailable"),
        "other_hover_total": sum(
            reason not in {"no_valid_candidate", "neighbor_conflict", "planner_unavailable"}
            for reason in flattened_hover_reasons
        ),
        "goal_conflict_resolution_total": int(sum(getattr(trace, "goal_conflict_resolutions", []))),
        "late_reassignment_steps": int(sum(getattr(trace, "late_reassignment_applied", []))),
        "physical_links_mean": _mean(trace.physical_communication_links),
        "effective_links_mean": _mean(trace.effective_communication_links),
        "global_sync_steps": int(sum(trace.global_sync_applied)),
        "active_agents_final": int(sum(trace.active_flags[-1])) if trace.active_flags else 0,
    }


def save_step_diagnostics(path: str | Path, trace) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(len(trace.coverage_curve)):
        hover_reasons = trace.hover_reasons[step] if step < len(trace.hover_reasons) else []
        goal_conflict_resolutions = (
            trace.goal_conflict_resolutions[step]
            if step < len(getattr(trace, "goal_conflict_resolutions", []))
            else 0
        )
        rows.append(
            {
                "step": step,
                "coverage_ratio": trace.coverage_curve[step],
                "team_new_coverage": trace.team_new_coverage[step],
                "repeated_coverage_ratio": trace.repeated_coverage_ratio[step],
                "communication_links": trace.communication_links[step],
                "physical_communication_links": trace.physical_communication_links[step],
                "effective_communication_links": trace.effective_communication_links[step],
                "global_sync_applied": trace.global_sync_applied[step],
                "collision_count": trace.collision_count[step],
                "frontier_count": trace.frontier_counts[step],
                "hover_count": trace.hover_counts[step],
                "hover_reasons": "|".join(hover_reasons),
                "goal_conflict_resolutions": goal_conflict_resolutions,
                "late_reassignment_applied": (
                    trace.late_reassignment_applied[step]
                    if step < len(getattr(trace, "late_reassignment_applied", []))
                    else False
                ),
                "adjusted_count": trace.adjusted_counts[step],
                "planner_failure_count": trace.planner_failure_counts[step],
                "zero_gain_streak": trace.zero_gain_streaks[step],
                "planner_statuses": "|".join(trace.planner_statuses[step]),
                "shield_statuses": "|".join(trace.shield_statuses[step]),
                "active_flags": "|".join(str(value) for value in trace.active_flags[step]),
            }
        )
    _write_dict_rows(destination, rows)


def save_failure_summaries(path: str | Path, summaries: Iterable, traces: Iterable) -> None:
    rows = []
    for episode_index, (summary, trace) in enumerate(zip(summaries, traces, strict=True), start=1):
        rows.append(
            {
                "episode": episode_index,
                **asdict(summary),
                **summarize_trace_diagnostics(trace),
            }
        )
    _write_dict_rows(Path(path), rows)


def _first_positive_step(values: list[int]) -> int | None:
    return next((index for index, value in enumerate(values) if value > 0), None)


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _write_dict_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
