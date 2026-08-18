#!/usr/bin/env python3
"""Summarize dynamic first-crossing episodes and the fixed T-20 timing bound."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rate(values: Iterable[bool]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _distribution(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if value is not None])
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def summarize_dynamic_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    selected = Counter(str(row["selected_option"]) for row in rows)
    base_success = [row for row in rows if row["reference_base_outcome"] == "task_success"]
    recoverable_t20 = [
        row for row in rows
        if row["reference_base_outcome"] == "catastrophe"
        and row["t20_oracle_timing_upper_bound"]["option_outcomes"][
            "detour_complete"
        ] == "task_success"
    ]
    interventions = [row for row in rows if row["intervened"]]
    return {
        "n_episodes": len(rows),
        "n_independent_source_states": len({
            str(row["source_state_sha256"]) for row in rows
        }),
        "task_success_rate": _rate(row["outcome"] == "task_success" for row in rows),
        "catastrophe_rate": _rate(row["outcome"] == "catastrophe" for row in rows),
        "safe_noncompletion_rate": _rate(
            row["outcome"] == "safe_noncompletion" for row in rows
        ),
        "intervention_rate": _rate(bool(row["intervened"]) for row in rows),
        "selected_option_counts": {
            option: int(selected[option])
            for option in ("base_continue", "detour_complete", "retreat_hold")
        },
        "first_trigger_lead_time_actions": _distribution(
            row["first_trigger_lead_time_actions"] for row in rows
        ),
        "first_trigger_to_collision_eef_distance_m": _distribution(
            row["first_trigger_to_collision_eef_distance_m"] for row in rows
        ),
        "missed_recovery_window_rate": _rate(
            bool(row["missed_recovery_window"]) for row in rows
        ),
        "missed_recovery_window_given_t20_recoverable": _rate(
            bool(row["missed_recovery_window"]) for row in recoverable_t20
        ),
        "n_t20_recoverable": len(recoverable_t20),
        "unnecessary_early_intervention_rate": _rate(
            bool(row["unnecessary_early_intervention"]) for row in rows
        ),
        "unnecessary_early_intervention_given_base_success": _rate(
            bool(row["unnecessary_early_intervention"]) for row in base_success
        ),
        "n_base_success": len(base_success),
        "intervention_duration_actions": _distribution(
            row["intervention_duration_actions"] for row in interventions
        ),
        "episode_peak_contact_force_p95_n": (
            float(np.percentile([
                float(row["contact_force_max_n"]) for row in rows
            ], 95)) if rows else None
        ),
        "episode_peak_contact_force_max_n": (
            float(max(float(row["contact_force_max_n"]) for row in rows))
            if rows else None
        ),
        "within_episode_contact_force_p95_n": _distribution(
            row["contact_force_p95_n"] for row in rows
        ),
    }


def summarize_fixed_t20(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    outcomes = [
        row["t20_oracle_timing_upper_bound"]["fixed_router_outcome"] for row in rows
    ]
    options = [
        row["t20_oracle_timing_upper_bound"]["fixed_router_option"] for row in rows
    ]
    counts = Counter(options)
    return {
        "role": "oracle_timing_upper_bound_not_deployable_method",
        "n_episodes": len(rows),
        "task_success_rate": _rate(outcome == "task_success" for outcome in outcomes),
        "catastrophe_rate": _rate(outcome == "catastrophe" for outcome in outcomes),
        "safe_noncompletion_rate": _rate(
            outcome == "safe_noncompletion" for outcome in outcomes
        ),
        "intervention_rate": _rate(option != "base_continue" for option in options),
        "selected_option_counts": {
            option: int(counts[option])
            for option in ("base_continue", "detour_complete", "retreat_hold")
        },
    }


def _write_episode_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = (
        "episode_id", "placement_id", "source_state_sha256", "condition",
        "reference_base_outcome", "trigger_action_index", "selected_option",
        "first_trigger_lead_time_actions",
        "first_trigger_to_collision_eef_distance_m", "missed_recovery_window",
        "unnecessary_early_intervention", "outcome", "intervention_duration_actions",
        "contact_force_p95_n", "contact_force_max_n",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _write_report(analysis: Mapping[str, Any], path: Path) -> None:
    dynamic = analysis["dynamic_first_crossing"]
    fixed = analysis["fixed_t20_oracle_timing_upper_bound"]
    lead = dynamic["first_trigger_lead_time_actions"]
    distance = dynamic["first_trigger_to_collision_eef_distance_m"]
    duration = dynamic["intervention_duration_actions"]
    lines = [
        "# P2 dynamic first-crossing Router",
        "",
        "The primary method scores from reset and latches the first non-Base option whose "
        "calibrated advantage lower score crosses zero. T-20 is reported only as a "
        "privileged timing upper bound.",
        "",
        "| Method | Task success | Catastrophe | Safe noncompletion | Intervention |",
        "|---|---:|---:|---:|---:|",
        f"| Dynamic first crossing | {_pct(dynamic['task_success_rate'])} | "
        f"{_pct(dynamic['catastrophe_rate'])} | {_pct(dynamic['safe_noncompletion_rate'])} | "
        f"{_pct(dynamic['intervention_rate'])} |",
        f"| Fixed T-20 (oracle timing) | {_pct(fixed['task_success_rate'])} | "
        f"{_pct(fixed['catastrophe_rate'])} | {_pct(fixed['safe_noncompletion_rate'])} | "
        f"{_pct(fixed['intervention_rate'])} |",
        "",
        "## Dynamic timing and intervention diagnostics",
        "",
        f"- Selected options: `{dynamic['selected_option_counts']}`.",
        f"- Intervention lead time: n={lead['n']}, median={lead['median']}, "
        f"mean={lead['mean']} actions.",
        f"- First-trigger distance to matched Base collision: n={distance['n']}, "
        f"median={distance['median']} m, mean={distance['mean']} m.",
        f"- Missed recovery window: {_pct(dynamic['missed_recovery_window_rate'])} overall; "
        f"{_pct(dynamic['missed_recovery_window_given_t20_recoverable'])} among "
        f"{dynamic['n_t20_recoverable']} T-20-recoverable episodes.",
        f"- Unnecessary early intervention: "
        f"{_pct(dynamic['unnecessary_early_intervention_rate'])} overall; "
        f"{_pct(dynamic['unnecessary_early_intervention_given_base_success'])} among "
        f"{dynamic['n_base_success']} Base-success episodes.",
        f"- Intervention duration: n={duration['n']}, median={duration['median']}, "
        f"p95={duration['p95']} actions.",
        f"- Episode peak contact force: p95="
        f"{dynamic['episode_peak_contact_force_p95_n']} N, max="
        f"{dynamic['episode_peak_contact_force_max_n']} N.",
        "",
        "Rates use episodes for transparency; the independent statistical unit is source state.",
    ]
    path.write_text("\n".join(lines) + "\n")


def analyze(capture: Path) -> dict[str, Any]:
    rows = _read_jsonl(capture / "dynamic_episodes.jsonl")
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    by_condition = {
        condition: summarize_dynamic_rows([
            row for row in rows if row["condition"] == condition
        ])
        for condition in sorted({str(row["condition"]) for row in rows})
    }
    analysis = {
        "schema_version": 1,
        "kind": "dynamic_first_crossing_counterfactual_router_analysis",
        "capture": str(capture),
        "capture_repository": manifest["repository"],
        "router_point": manifest["router_point"],
        "dynamic_first_crossing": summarize_dynamic_rows(rows),
        "fixed_t20_oracle_timing_upper_bound": summarize_fixed_t20(rows),
        "dynamic_by_condition": by_condition,
    }
    (capture / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    _write_episode_csv(rows, capture / "episodes.csv")
    _write_report(analysis, capture / "REPORT.md")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.capture.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
