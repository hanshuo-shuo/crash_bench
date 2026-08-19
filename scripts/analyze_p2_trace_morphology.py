#!/usr/bin/env python3
"""Audit temporal morphology of the frozen P2 Router score trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OPTIONS = ("base_continue", "detour_complete", "retreat_hold")
SIGNALS = ("detour", "retreat_hold", "max_nonbase")
PRIMARY_GROUPS = (
    "missed_t20_recoverable_glass",
    "successfully_recovered_glass",
    "other_glass",
    "base_success_offpath_noglass_control",
)
GROUP_COLORS = {
    "missed_t20_recoverable_glass": (201, 54, 46),
    "successfully_recovered_glass": (18, 128, 87),
    "other_glass": (102, 102, 102),
    "base_success_offpath_noglass_control": (43, 116, 189),
    "outside_requested_groups": (155, 155, 155),
}
RANKING_METRICS = (
    ("max_nonbase_raw_max", "Raw max"),
    ("max_nonbase_moving_average_max_k3", "MA-3 max"),
    ("max_nonbase_moving_average_max_k5", "MA-5 max"),
    ("max_nonbase_moving_average_max_k8", "MA-8 max"),
    ("max_nonbase_top5_mean", "Top-5 mean"),
    (
        "max_nonbase_cumulative_excess_above_pointwise_margin",
        "Excess area",
    ),
    (
        "max_nonbase_longest_run_above_pointwise_margin",
        "Longest > margin",
    ),
    ("longest_stable_nonbase_candidate_run", "Candidate run"),
    ("max_nonbase_trend_slope_last10_pre_event", "Last-10 slope"),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def moving_average_max(values: Sequence[float], window: int) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if window <= 0:
        raise ValueError("moving-average window must be positive")
    if len(array) < window:
        return None
    means = np.convolve(array, np.ones(window) / window, mode="valid")
    return float(np.max(means))


def cumulative_area_above(values: Sequence[float], margin: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.maximum(array - float(margin), 0.0).sum())


def longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def longest_value_run(values: Sequence[str], target: str) -> int:
    return longest_true_run(value == target for value in values)


def episode_group(row: Mapping[str, Any]) -> str:
    if row["condition"] == "glass":
        if bool(row["missed_recovery_window"]):
            return "missed_t20_recoverable_glass"
        if (
            row["reference_base_outcome"] == "catastrophe"
            and row["outcome"] == "task_success"
            and bool(row["intervened"])
        ):
            return "successfully_recovered_glass"
        return "other_glass"
    if (
        row["condition"] in {"offpath", "noglass"}
        and row["reference_base_outcome"] == "task_success"
    ):
        return "base_success_offpath_noglass_control"
    return "outside_requested_groups"


def _trend_slope(
    actions: Sequence[int],
    values: Sequence[float],
    *,
    anchor_action: int,
    window: int = 10,
) -> tuple[float | None, int]:
    selected = [
        (int(action), float(value))
        for action, value in zip(actions, values)
        if int(action) <= int(anchor_action)
    ][-window:]
    if len(selected) < 2:
        return None, len(selected)
    x = np.asarray([item[0] for item in selected], dtype=np.float64)
    y = np.asarray([item[1] for item in selected], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0]), len(selected)


def _signal_statistics(
    values: Sequence[float],
    *,
    actions: Sequence[int],
    pointwise_margin: float,
    anchor_action: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    maximum_index = int(np.argmax(array))
    slope, slope_points = _trend_slope(
        actions, array, anchor_action=anchor_action, window=10
    )
    return {
        "raw_max": float(array[maximum_index]),
        "raw_max_action_index": int(actions[maximum_index]),
        "moving_average_max_k3": moving_average_max(array, 3),
        "moving_average_max_k5": moving_average_max(array, 5),
        "moving_average_max_k8": moving_average_max(array, 8),
        "top5_mean": float(np.mean(np.sort(array)[-min(5, len(array)):])),
        "cumulative_excess_above_pointwise_margin": cumulative_area_above(
            array, pointwise_margin
        ),
        "longest_run_above_pointwise_margin": longest_true_run(
            array > pointwise_margin
        ),
        "count_above_pointwise_margin": int(np.sum(array > pointwise_margin)),
        "fraction_above_pointwise_margin": float(
            np.mean(array > pointwise_margin)
        ),
        "trend_slope_last10_pre_event": slope,
        "trend_points": slope_points,
    }


def _flatten_signal_statistics(
    result: dict[str, Any], signal: str, statistics: Mapping[str, Any]
) -> None:
    for name, value in statistics.items():
        result[f"{signal}_{name}"] = value


def _alignment_anchor(row: Mapping[str, Any], last_action: int) -> tuple[int, str]:
    if row.get("trigger_action_index") is not None:
        return int(row["trigger_action_index"]), "trigger"
    if row.get("reference_collision_action_index") is not None:
        return int(row["reference_collision_action_index"]), "collision"
    return int(last_action), "trace_end_no_collision_or_trigger"


def _window_statistics(
    reconstructed: Mapping[str, Any], episode: Mapping[str, Any], margin: float
) -> dict[str, Any] | None:
    if episode["condition"] != "glass":
        return None
    start = int(episode["t20_oracle_timing_upper_bound"]["anchor_action_index"])
    end_value = episode.get("reference_collision_action_index")
    if end_value is None:
        return None
    end = int(end_value)
    indices = [
        index
        for index, action in enumerate(reconstructed["action_index"])
        if start <= int(action) <= end
    ]
    if not indices:
        return None
    actions = [int(reconstructed["action_index"][index]) for index in indices]
    expected = end - start + 1
    result: dict[str, Any] = {
        "start_action_index": start,
        "end_action_index": end,
        "expected_actions": expected,
        "observed_actions": len(indices),
        "complete": len(indices) == expected and actions == list(range(start, end + 1)),
    }
    for signal in SIGNALS:
        values = [float(reconstructed[f"{signal}_advantage"][index]) for index in indices]
        stats = _signal_statistics(
            values,
            actions=actions,
            pointwise_margin=margin,
            anchor_action=end,
        )
        _flatten_signal_statistics(result, signal, stats)
    candidates = [reconstructed["candidate_option"][index] for index in indices]
    result.update({
        "detour_candidate_count": int(sum(
            option == "detour_complete" for option in candidates
        )),
        "detour_candidate_fraction": float(np.mean([
            option == "detour_complete" for option in candidates
        ])),
        "longest_detour_candidate_run": longest_value_run(
            candidates, "detour_complete"
        ),
    })
    return result


def _reconstruct_trace(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["action_index"]))
    actions = [int(row["action_index"]) for row in ordered]
    if actions != list(range(actions[0], actions[-1] + 1)):
        raise ValueError("router trace action indices are not contiguous")
    advantage = np.asarray(
        [row["advantage_vs_base"] for row in ordered], dtype=np.float64
    )
    if advantage.ndim != 2 or advantage.shape[1] != len(OPTIONS):
        raise ValueError("expected three option advantages per trace row")
    detour = advantage[:, 1]
    retreat = advantage[:, 2]
    candidate_indices = 1 + np.argmax(advantage[:, 1:], axis=1)
    candidates = [OPTIONS[int(index)] for index in candidate_indices]
    recorded_candidates = [str(row["candidate_option"]) for row in ordered]
    if candidates != recorded_candidates:
        raise ValueError("candidate option sequence disagrees with reconstructed argmax")
    option_argmax = [OPTIONS[int(index)] for index in np.argmax(advantage, axis=1)]
    return {
        "action_index": actions,
        "detour_advantage": detour.tolist(),
        "retreat_hold_advantage": retreat.tolist(),
        "max_nonbase_advantage": np.maximum(detour, retreat).tolist(),
        "candidate_option": candidates,
        "option_argmax": option_argmax,
        "first_crossing": [bool(row["first_crossing"]) for row in ordered],
    }


def _episode_statistics(
    episode: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    pointwise_margin: float,
    sequential_margin: float,
    cohort: str,
) -> dict[str, Any]:
    trace = _reconstruct_trace(trace_rows)
    actions = trace["action_index"]
    anchor, anchor_type = _alignment_anchor(episode, actions[-1])
    result: dict[str, Any] = {
        "episode_id": str(episode["episode_id"]),
        "placement_id": str(episode["placement_id"]),
        "source_state_sha256": str(episode["source_state_sha256"]),
        "condition": str(episode["condition"]),
        "group": episode_group(episode),
        "cohort": cohort,
        "reference_base_outcome": str(episode["reference_base_outcome"]),
        "dynamic_outcome": str(episode["outcome"]),
        "intervened": bool(episode["intervened"]),
        "selected_option": str(episode["selected_option"]),
        "missed_recovery_window": bool(episode["missed_recovery_window"]),
        "trace_actions": len(actions),
        "trace_first_action_index": actions[0],
        "trace_last_action_index": actions[-1],
        "reference_collision_action_index": episode.get(
            "reference_collision_action_index"
        ),
        "trigger_action_index": episode.get("trigger_action_index"),
        "alignment_anchor_action_index": anchor,
        "alignment_anchor_type": anchor_type,
        "pointwise_margin": float(pointwise_margin),
        "sequential_margin": float(sequential_margin),
    }
    for signal in SIGNALS:
        stats = _signal_statistics(
            trace[f"{signal}_advantage"],
            actions=actions,
            pointwise_margin=pointwise_margin,
            anchor_action=anchor,
        )
        _flatten_signal_statistics(result, signal, stats)

    candidates = trace["candidate_option"]
    option_argmax = trace["option_argmax"]
    detour_candidate_run = longest_value_run(candidates, "detour_complete")
    retreat_candidate_run = longest_value_run(candidates, "retreat_hold")
    result.update({
        "max_nonbase_raw_max_option": candidates[int(np.argmax(
            trace["max_nonbase_advantage"]
        ))],
        "longest_detour_candidate_run": detour_candidate_run,
        "longest_retreat_hold_candidate_run": retreat_candidate_run,
        "longest_stable_nonbase_candidate_run": max(
            detour_candidate_run, retreat_candidate_run
        ),
        "longest_detour_option_argmax_run": longest_value_run(
            option_argmax, "detour_complete"
        ),
        "longest_retreat_hold_option_argmax_run": longest_value_run(
            option_argmax, "retreat_hold"
        ),
        "recovery_window": _window_statistics(trace, episode, pointwise_margin),
        "trajectory": {
            **trace,
            "aligned_action_index": [action - anchor for action in actions],
        },
    })
    return result


def _load_episode_statistics(
    router_trace: Path,
    dynamic_episodes: Path,
    *,
    pointwise_margin: float,
    sequential_margin: float,
    cohort: str,
    placement_id: str | None = None,
) -> list[dict[str, Any]]:
    episodes = _read_jsonl(dynamic_episodes)
    if placement_id is not None:
        episodes = [row for row in episodes if row["placement_id"] == placement_id]
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(router_trace):
        by_episode[str(row["episode_id"])].append(row)
    results = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        if episode_id not in by_episode:
            raise ValueError(f"missing router trace for episode {episode_id}")
        results.append(_episode_statistics(
            episode,
            by_episode[episode_id],
            pointwise_margin=pointwise_margin,
            sequential_margin=sequential_margin,
            cohort=cohort,
        ))
    return results


def _rank(values: Sequence[float], value: float) -> int:
    return 1 + sum(candidate > value for candidate in values)


def _pairwise_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    comparisons = [
        1.0 if pos > neg else 0.5 if pos == neg else 0.0
        for pos in positive
        for neg in negative
    ]
    return float(np.mean(comparisons))


def _rankings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row["group"] in PRIMARY_GROUPS]
    missed_control = [
        row for row in primary
        if row["group"] in {
            "missed_t20_recoverable_glass",
            "base_success_offpath_noglass_control",
        }
    ]
    recoverable_control = [
        row for row in primary
        if row["group"] in {
            "missed_t20_recoverable_glass",
            "successfully_recovered_glass",
            "base_success_offpath_noglass_control",
        }
    ]
    result: dict[str, Any] = {}
    for metric, label in RANKING_METRICS:
        ordered = sorted(
            primary,
            key=lambda row: (-float(row[metric]), row["placement_id"], row["condition"]),
        )
        primary_values = [float(row[metric]) for row in primary]
        comparison_values = [float(row[metric]) for row in missed_control]
        entries = []
        for position, row in enumerate(ordered, start=1):
            value = float(row[metric])
            entry = {
                "position": position,
                "rank": _rank(primary_values, value),
                "tie_size": sum(candidate == value for candidate in primary_values),
                "episode_id": row["episode_id"],
                "placement_id": row["placement_id"],
                "condition": row["condition"],
                "group": row["group"],
                "value": value,
            }
            if row in missed_control:
                entry["missed_control_rank"] = _rank(comparison_values, value)
            entries.append(entry)
        missed = [
            float(row[metric]) for row in primary
            if row["group"] == "missed_t20_recoverable_glass"
        ]
        recovered = [
            float(row[metric]) for row in primary
            if row["group"] == "successfully_recovered_glass"
        ]
        controls = [
            float(row[metric]) for row in primary
            if row["group"] == "base_success_offpath_noglass_control"
        ]
        result[metric] = {
            "label": label,
            "descending_higher_is_more_evidence": True,
            "entries": entries,
            "missed_vs_control_auc": _pairwise_auc(missed, controls),
            "all_recoverable_treatment_vs_control_auc": _pairwise_auc(
                missed + recovered, controls
            ),
        }
    return result


def _group_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for group in PRIMARY_GROUPS:
        members = [row for row in rows if row["group"] == group]
        metrics = {}
        for metric, _ in RANKING_METRICS:
            values = [float(row[metric]) for row in members]
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "median": float(median(values)),
                "min": float(min(values)),
                "max": float(max(values)),
            }
        result[group] = {"n": len(members), "metrics": metrics}
    return result


def _input_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.capture_manifest.read_text())
    boundary = json.loads(args.sequential_boundary.read_text())
    pointwise_margin = float(boundary["boundaries"][
        f"alpha_{float(boundary['primary_alpha']):.1f}"
    ]["pointwise_margin"])
    sequential_margin = float(boundary["boundaries"][
        f"alpha_{float(boundary['primary_alpha']):.1f}"
    ]["effective_margin"])
    router_point = manifest["router_point"]
    if not np.isclose(float(router_point["pointwise_margin"]), pointwise_margin):
        raise ValueError("capture and sequential boundary pointwise margins disagree")
    if not np.isclose(float(router_point["effective_margin"]), sequential_margin):
        raise ValueError("capture and sequential boundary effective margins disagree")

    rows = _load_episode_statistics(
        args.router_trace,
        args.dynamic_episodes,
        pointwise_margin=pointwise_margin,
        sequential_margin=sequential_margin,
        cohort="stable_four_source",
    )
    unstable_rows: list[dict[str, Any]] = []
    unstable_inputs = None
    unstable_args = (
        args.unstable_router_trace,
        args.unstable_dynamic_episodes,
        args.unstable_capture_manifest,
    )
    if any(item is not None for item in unstable_args):
        if not all(item is not None for item in unstable_args):
            raise ValueError("all three unstable capture paths are required together")
        unstable_manifest = json.loads(args.unstable_capture_manifest.read_text())
        unstable_rows = _load_episode_statistics(
            args.unstable_router_trace,
            args.unstable_dynamic_episodes,
            pointwise_margin=pointwise_margin,
            sequential_margin=sequential_margin,
            cohort="unstable_diagnostic_not_in_main_table",
            placement_id=args.unstable_placement_id,
        )
        unstable_inputs = {
            "router_trace": _input_record(args.unstable_router_trace),
            "dynamic_episodes": _input_record(args.unstable_dynamic_episodes),
            "capture_manifest": _input_record(args.unstable_capture_manifest),
            "capture_repository": unstable_manifest["repository"],
            "router_point": unstable_manifest["router_point"],
            "placement_filter": args.unstable_placement_id,
        }

    ranking = _rankings(rows)
    result = {
        "schema_version": 1,
        "kind": "p2_dynamic_router_trace_morphology_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "primary_score": "max(Detour advantage, Retreat/Hold advantage)",
            "moving_average": "trailing-free valid-window mean; reported statistic is its maximum",
            "top5_mean": "mean of the five largest action scores",
            "area": "discrete sum max(score - pointwise_margin, 0)",
            "longest_above_margin": "strictly score > pointwise_margin",
            "candidate_run": "longest unchanged best non-Base option, even when Base is overall argmax",
            "trend": "OLS slope over the last up-to-10 trace actions ending at trigger, collision, or trace end for no-event controls",
            "ranking": "descending competition rank among the four requested stable groups only",
        },
        "margins": {
            "pointwise": pointwise_margin,
            "sequential_effective": sequential_margin,
        },
        "inputs": {
            "router_trace": _input_record(args.router_trace),
            "dynamic_episodes": _input_record(args.dynamic_episodes),
            "capture_manifest": _input_record(args.capture_manifest),
            "sequential_boundary": _input_record(args.sequential_boundary),
            "capture_repository": manifest["repository"],
            "router_point": router_point,
            "unstable": unstable_inputs,
        },
        "cohort": {
            "stable_source_count": len({row["source_state_sha256"] for row in rows}),
            "stable_episode_count": len(rows),
            "primary_comparison_episode_count": sum(
                row["group"] in PRIMARY_GROUPS for row in rows
            ),
            "outside_requested_groups": [
                row["episode_id"] for row in rows
                if row["group"] == "outside_requested_groups"
            ],
            "unstable_diagnostic_episode_count": len(unstable_rows),
        },
        "group_summaries": _group_summaries(rows),
        "rankings": ranking,
        "episodes": rows,
        "unstable_diagnostic": unstable_rows,
    }
    return result


def _csv_rows(analysis: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    excluded = {"trajectory", "recovery_window"}
    rows = []
    for source in (analysis["episodes"], analysis["unstable_diagnostic"]):
        for row in source:
            flat = {key: value for key, value in row.items() if key not in excluded}
            window = row.get("recovery_window")
            if window is not None:
                flat.update({f"t20_window_{key}": value for key, value in window.items()})
            rows.append(flat)
    preferred = [
        "episode_id", "placement_id", "source_state_sha256", "condition", "group",
        "cohort", "reference_base_outcome", "dynamic_outcome", "intervened",
        "selected_option", "missed_recovery_window", "trace_actions",
        "reference_collision_action_index", "trigger_action_index",
        "alignment_anchor_action_index", "alignment_anchor_type",
    ]
    keys = {key for row in rows for key in row}
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))
    return fields, rows


def _write_csv(analysis: Mapping[str, Any], path: Path) -> None:
    fields, rows = _csv_rows(analysis)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _rank_entry(
    analysis: Mapping[str, Any], metric: str, placement_id: str
) -> Mapping[str, Any]:
    return next(
        entry for entry in analysis["rankings"][metric]["entries"]
        if entry["placement_id"] == placement_id and entry["condition"] == "glass"
    )


def _write_report(analysis: Mapping[str, Any], path: Path) -> None:
    rows = analysis["episodes"]
    missed = sorted(
        [row for row in rows if row["group"] == "missed_t20_recoverable_glass"],
        key=lambda row: row["placement_id"],
    )
    controls = [
        row for row in rows
        if row["group"] == "base_success_offpath_noglass_control"
    ]
    raw_metric = "max_nonbase_raw_max"
    temporal_metrics = [metric for metric, _ in RANKING_METRICS if metric != raw_metric]
    best_auc = max(
        analysis["rankings"][metric]["missed_vs_control_auc"]
        for metric in temporal_metrics
    )
    best_labels = [
        analysis["rankings"][metric]["label"] for metric in temporal_metrics
        if np.isclose(
            analysis["rankings"][metric]["missed_vs_control_auc"], best_auc
        )
    ]
    raw_auc = analysis["rankings"][raw_metric]["missed_vs_control_auc"]

    lines = [
        "# P2 Dynamic Router trace morphology audit",
        "",
        "**Finding:** simple temporal aggregation does not repair the treatment/control "
        "ordering. It partially promotes `heldout_0010`, whose Detour signal is "
        "sustained, but demotes `heldout_0019`, whose evidence is a short burst. "
        "Several benign controls also carry sustained high scores.",
        "",
        "This is a diagnostic analysis of existing artifacts only: no model training, "
        "new rollout, or instantaneous-threshold search was performed.",
        "",
        "## Cohort and definitions",
        "",
        f"- Stable cohort: {analysis['cohort']['stable_source_count']} sources, "
        f"{analysis['cohort']['stable_episode_count']} episodes; "
        f"{analysis['cohort']['primary_comparison_episode_count']} episodes belong to "
        "the four requested groups.",
        "- The remaining stable episode is `heldout_0019/noglass`, whose matched Base "
        "outcome is safe noncompletion. Its episode statistics are preserved, but it "
        "is not a Base-success control and is excluded from ranks and AUCs.",
        f"- Primary score: max non-Base advantage. Pointwise margin: "
        f"`{analysis['margins']['pointwise']:.10f}`; frozen sequential margin: "
        f"`{analysis['margins']['sequential_effective']:.10f}`.",
        "- Trend is fitted to the last up-to-10 actions ending at trigger or collision; "
        "a no-event control uses trace end.",
        "",
        "## Ranking of the two missed recoverable episodes",
        "",
        "Ranks are descending among the 11 stable episodes in the four requested "
        "groups. Parentheses give rank among the two missed treatments plus seven "
        "Base-success controls.",
        "",
        "| Statistic | heldout_0010 | heldout_0019 | Missed-v-control AUC | All recoverable-v-control AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, label in RANKING_METRICS:
        entries = [_rank_entry(analysis, metric, row["placement_id"]) for row in missed]
        display = []
        for entry in entries:
            tie = f", tie {entry['tie_size']}" if entry["tie_size"] > 1 else ""
            display.append(
                f"{entry['rank']}/11 ({entry['missed_control_rank']}/9{tie}); "
                f"{_fmt(entry['value'])}"
            )
        ranking = analysis["rankings"][metric]
        lines.append(
            f"| {label} | {display[0]} | {display[1]} | "
            f"{ranking['missed_vs_control_auc']:.3f} | "
            f"{ranking['all_recoverable_treatment_vs_control_auc']:.3f} |"
        )

    lines.extend([
        "",
        f"Raw maximum gives missed-v-control AUC `{raw_auc:.3f}`. The best temporal "
        f"alternatives ({', '.join(best_labels)}) reach only `{best_auc:.3f}`. No "
        "listed temporal statistic improves the pairwise ordering of both missed "
        "episodes.",
        "",
        "## Are the control highs only one-step spikes?",
        "",
        "No. The strongest no-glass control (`heldout_0008`) is spike-heavy, but "
        "other false positives remain high after smoothing and accumulation.",
        "",
        "| Control | Raw max | MA-5 max | MA-5/raw | Excess area | Longest > margin |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(controls, key=lambda item: (item["placement_id"], item["condition"])):
        ratio = row["max_nonbase_moving_average_max_k5"] / row["max_nonbase_raw_max"]
        lines.append(
            f"| {row['placement_id'].removeprefix('glass_recovery_')}/{row['condition']} | "
            f"{row['max_nonbase_raw_max']:.3f} | "
            f"{row['max_nonbase_moving_average_max_k5']:.3f} | {ratio:.3f} | "
            f"{row['max_nonbase_cumulative_excess_above_pointwise_margin']:.2f} | "
            f"{row['max_nonbase_longest_run_above_pointwise_margin']} |"
        )
    high_0010_raw = sum(
        row["max_nonbase_raw_max"] > missed[0]["max_nonbase_raw_max"]
        for row in controls
    )
    high_0010_ma5 = sum(
        row["max_nonbase_moving_average_max_k5"]
        > missed[0]["max_nonbase_moving_average_max_k5"]
        for row in controls
    )
    high_0019_raw = sum(
        row["max_nonbase_raw_max"] > missed[1]["max_nonbase_raw_max"]
        for row in controls
    )
    high_0019_ma5 = sum(
        row["max_nonbase_moving_average_max_k5"]
        > missed[1]["max_nonbase_moving_average_max_k5"]
        for row in controls
    )
    lines.extend([
        "",
        f"Smoothing helps only locally: controls above `heldout_0010` fall from "
        f"{high_0010_raw}/7 under raw max to {high_0010_ma5}/7 under MA-5. For "
        f"`heldout_0019`, they increase from {high_0019_raw}/7 to "
        f"{high_0019_ma5}/7 because its own maximum is less persistent.",
        "",
        "## Detour evidence in the T-20 window",
        "",
        "| Episode | Detour max | Detour actions > margin | Longest Detour > margin | Detour candidate actions | Longest Detour-candidate run |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in missed:
        window = row["recovery_window"]
        lines.append(
            f"| {row['placement_id'].removeprefix('glass_recovery_')} | "
            f"{window['detour_raw_max']:.3f} | "
            f"{window['detour_count_above_pointwise_margin']}/20 | "
            f"{window['detour_longest_run_above_pointwise_margin']} | "
            f"{window['detour_candidate_count']}/20 | "
            f"{window['longest_detour_candidate_run']} |"
        )
    lines.extend([
        "",
        "`heldout_0010` contains a real sustained Detour segment: 11 consecutive "
        "actions above the pointwise margin and Detour is the best non-Base option "
        "for 18/20 actions. `heldout_0019` has only a five-action Detour-above-margin "
        "burst. Its 17/20 Detour candidate count is not strong positive evidence: the "
        "candidate comparison ignores Base and often just says Retreat is worse.",
        "",
        "## Diagnostic decision",
        "",
        "1. **Simple evidence accumulator:** do not freeze one now. MA-3/5/8, top-5, "
        "area, longest run, candidate stability, and last-10 slope all fail to improve "
        "the two-missed-v-control AUC over raw maximum. Area and run length are "
        "especially confounded by long benign trajectories.",
        "2. **Recovery-window supervision:** this is the minimum justified next "
        "experiment. Train or calibrate Detour intervention value against explicit "
        "action-level recoverability windows, freeze once, and evaluate on a new "
        "source-disjoint cohort. The present score can be temporally persistent while "
        "still semantically wrong on controls.",
        "3. **Temporal value model:** defer it. It is a larger change and these 12 "
        "episodes do not show that a learned sequence model is needed before fixing "
        "the supervision target.",
        "",
        "This is not a new project-level GO/NO-GO. The partial result is that smoothing "
        "removes one spike-driven control and improves `heldout_0010`'s rank, but the "
        "failure mode is heterogeneous and no single simple statistic covers both "
        "recoverable treatments.",
    ])

    unstable = analysis["unstable_diagnostic"]
    if unstable:
        lines.extend([
            "",
            "## Unstable heldout_0015 diagnostic (excluded from all main results)",
            "",
            "This older capture used the pointwise first-crossing margin, not the "
            "source-level sequential boundary. All three traces end at their early "
            "trigger, and the source was node-unstable, so these values are descriptive "
            "only and are not added to ranks or AUCs.",
            "",
            "| Condition | Trace actions | Trigger | Collision | Raw max | MA-5 max |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(unstable, key=lambda item: item["condition"]):
            lines.append(
                f"| {row['condition']} | {row['trace_actions']} | "
                f"{_fmt(row['trigger_action_index'])} | "
                f"{_fmt(row['reference_collision_action_index'])} | "
                f"{row['max_nonbase_raw_max']:.3f} | "
                f"{row['max_nonbase_moving_average_max_k5']:.3f} |"
            )

    lines.extend([
        "",
        "## Artifacts",
        "",
        "- Full per-action reconstructed sequences and rankings: "
        "`p2_trace_morphology_audit_20260819.json`.",
        "- Flat per-episode statistics: "
        "`p2_trace_morphology_episode_stats_20260819.csv`.",
        "- Trajectory and ranking figures: "
        "`p2_trace_morphology_trajectories_20260819.png` and "
        "`p2_trace_morphology_rankings_20260819.png`.",
    ])
    path.write_text("\n".join(lines) + "\n")


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    names = (
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def _draw_text(draw, xy: tuple[int, int], text: str, *, size: int, color=(20, 20, 20), bold=False, anchor=None):
    draw.text(xy, text, fill=color, font=_font(size, bold=bold), anchor=anchor)


def _episode_color(row: Mapping[str, Any]) -> tuple[int, int, int]:
    if row["group"] == "missed_t20_recoverable_glass":
        return (201, 54, 46) if row["placement_id"].endswith("0010") else (225, 123, 36)
    return GROUP_COLORS[row["group"]]


def _write_trajectory_figure(analysis: Mapping[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1800, 1120), "white")
    draw = ImageDraw.Draw(image)
    _draw_text(draw, (80, 45), "P2 score trajectories aligned to collision / trigger", size=34, bold=True)
    _draw_text(
        draw, (80, 92),
        "Stable four-source cohort; max non-Base advantage; last 50 actions before event",
        size=20, color=(75, 75, 75),
    )
    pointwise = float(analysis["margins"]["pointwise"])
    sequential = float(analysis["margins"]["sequential_effective"])
    panels = [
        ("Glass treatments", [
            row for row in analysis["episodes"]
            if row["group"] in {
                "missed_t20_recoverable_glass",
                "successfully_recovered_glass",
                "other_glass",
            }
        ]),
        ("Base-success off-path / no-glass controls", [
            row for row in analysis["episodes"]
            if row["group"] == "base_success_offpath_noglass_control"
        ]),
    ]
    left, right = 120, 1735
    y_min, y_max = -0.75, 1.78
    x_min, x_max = -50, 0
    for panel_index, (title, rows) in enumerate(panels):
        top = 155 + panel_index * 455
        bottom = top + 335
        _draw_text(draw, (left, top - 38), title, size=24, bold=True)
        for tick in (-0.5, 0.0, 0.5, 1.0, 1.5):
            y = int(bottom - (tick - y_min) / (y_max - y_min) * (bottom - top))
            draw.line((left, y, right, y), fill=(226, 226, 226), width=1)
            _draw_text(draw, (left - 16, y), f"{tick:.1f}", size=17, color=(80, 80, 80), anchor="rm")
        for tick in (-50, -40, -30, -20, -10, 0):
            x = int(left + (tick - x_min) / (x_max - x_min) * (right - left))
            draw.line((x, top, x, bottom), fill=(236, 236, 236), width=1)
            _draw_text(draw, (x, bottom + 12), str(tick), size=17, color=(80, 80, 80), anchor="ma")
        for margin, color, label in (
            (pointwise, (132, 92, 35), "pointwise"),
            (sequential, (100, 45, 120), "sequential"),
        ):
            y = int(bottom - (margin - y_min) / (y_max - y_min) * (bottom - top))
            draw.line((left, y, right, y), fill=color, width=3)
            _draw_text(draw, (right - 6, y - 5), label, size=15, color=color, anchor="rb")
        draw.line((right, top, right, bottom), fill=(30, 30, 30), width=3)
        for row_index, row in enumerate(rows):
            aligned = row["trajectory"]["aligned_action_index"]
            values = row["trajectory"]["max_nonbase_advantage"]
            points = []
            for x_value, y_value in zip(aligned, values):
                if x_min <= x_value <= x_max:
                    x = int(left + (x_value - x_min) / (x_max - x_min) * (right - left))
                    y = int(bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top))
                    points.append((x, y))
            color = _episode_color(row)
            if len(points) >= 2:
                draw.line(points, fill=color, width=5 if "glass" in row["group"] else 3)
            if points:
                draw.ellipse(
                    (points[-1][0] - 4, points[-1][1] - 4, points[-1][0] + 4, points[-1][1] + 4),
                    fill=color,
                )
            label = f"{row['placement_id'][-4:]}/{row['condition']}"
            legend_x = left + (row_index % 4) * 350
            legend_y = bottom + 55 + (row_index // 4) * 30
            draw.line((legend_x, legend_y + 9, legend_x + 32, legend_y + 9), fill=color, width=5)
            _draw_text(draw, (legend_x + 42, legend_y), label, size=17, color=color)
        _draw_text(draw, ((left + right) // 2, bottom + 100), "Actions relative to collision / trigger (0)", size=18, anchor="ma")
    image.save(path)


def _write_ranking_figure(analysis: Mapping[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw

    primary = [
        row for row in analysis["episodes"] if row["group"] in PRIMARY_GROUPS
    ]
    group_order = {group: index for index, group in enumerate(PRIMARY_GROUPS)}
    primary.sort(key=lambda row: (
        group_order[row["group"]], row["placement_id"], row["condition"]
    ))
    width, height = 2040, 1040
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _draw_text(draw, (70, 42), "Episode ranking under simple temporal statistics", size=34, bold=True)
    _draw_text(
        draw, (70, 88),
        "Descending rank among 11 episodes in the four requested groups (1 = strongest evidence)",
        size=20, color=(75, 75, 75),
    )
    label_left = 70
    grid_left = 465
    cell_width = 165
    grid_top = 210
    cell_height = 62
    wrapped_labels = {
        "Raw max": ("Raw", "max"),
        "MA-3 max": ("MA-3", "max"),
        "MA-5 max": ("MA-5", "max"),
        "MA-8 max": ("MA-8", "max"),
        "Top-5 mean": ("Top-5", "mean"),
        "Excess area": ("Excess", "area"),
        "Longest > margin": ("Longest", "> margin"),
        "Candidate run": ("Candidate", "run"),
        "Last-10 slope": ("Last-10", "slope"),
    }
    for column, (_, label) in enumerate(RANKING_METRICS):
        x = grid_left + column * cell_width + cell_width // 2
        for line_index, part in enumerate(wrapped_labels[label]):
            _draw_text(draw, (x, 150 + line_index * 23), part, size=17, bold=True, anchor="ma")
    rank_maps = {}
    for metric, _ in RANKING_METRICS:
        rank_maps[metric] = {
            entry["episode_id"]: entry["rank"]
            for entry in analysis["rankings"][metric]["entries"]
        }
    for row_index, row in enumerate(primary):
        y = grid_top + row_index * cell_height
        color = _episode_color(row)
        draw.ellipse((label_left, y + 19, label_left + 20, y + 39), fill=color)
        label = f"{row['placement_id'][-4:]} / {row['condition']}"
        _draw_text(draw, (label_left + 34, y + 14), label, size=20, bold="glass" in row["group"], color=color)
        for column, (metric, _) in enumerate(RANKING_METRICS):
            rank = rank_maps[metric][row["episode_id"]]
            fraction = (rank - 1) / 10
            fill = (
                int(40 + 200 * fraction),
                int(103 + 130 * fraction),
                int(173 + 65 * fraction),
            )
            x = grid_left + column * cell_width
            draw.rounded_rectangle(
                (x + 4, y + 4, x + cell_width - 5, y + cell_height - 5),
                radius=8, fill=fill,
            )
            text_color = "white" if rank <= 4 else (25, 25, 25)
            _draw_text(
                draw, (x + cell_width // 2, y + cell_height // 2),
                str(rank), size=23, bold=True, color=text_color, anchor="mm",
            )
    note_y = grid_top + len(primary) * cell_height + 30
    _draw_text(draw, (grid_left, note_y), "Darker cells = higher evidence rank", size=18, color=(75, 75, 75))
    image.save(path)


def _resolved(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-trace", required=True, type=_resolved)
    parser.add_argument("--dynamic-episodes", required=True, type=_resolved)
    parser.add_argument("--capture-manifest", required=True, type=_resolved)
    parser.add_argument("--sequential-boundary", required=True, type=_resolved)
    parser.add_argument("--output-report", required=True, type=_resolved)
    parser.add_argument("--output-csv", required=True, type=_resolved)
    parser.add_argument("--output-json", required=True, type=_resolved)
    parser.add_argument("--trajectory-figure", required=True, type=_resolved)
    parser.add_argument("--ranking-figure", required=True, type=_resolved)
    parser.add_argument("--unstable-router-trace", type=_resolved)
    parser.add_argument("--unstable-dynamic-episodes", type=_resolved)
    parser.add_argument("--unstable-capture-manifest", type=_resolved)
    parser.add_argument(
        "--unstable-placement-id", default="glass_recovery_heldout_0015"
    )
    args = parser.parse_args()
    for path in (
        args.output_report,
        args.output_csv,
        args.output_json,
        args.trajectory_figure,
        args.ranking_figure,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    analysis = analyze(args)
    args.output_json.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(analysis, args.output_csv)
    _write_report(analysis, args.output_report)
    _write_trajectory_figure(analysis, args.trajectory_figure)
    _write_ranking_figure(analysis, args.ranking_figure)
    print(json.dumps({
        "stable_episodes": analysis["cohort"]["stable_episode_count"],
        "primary_comparison_episodes": analysis["cohort"][
            "primary_comparison_episode_count"
        ],
        "unstable_diagnostic_episodes": analysis["cohort"][
            "unstable_diagnostic_episode_count"
        ],
        "outputs": {
            "report": str(args.output_report),
            "csv": str(args.output_csv),
            "json": str(args.output_json),
            "trajectory_figure": str(args.trajectory_figure),
            "ranking_figure": str(args.ranking_figure),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
