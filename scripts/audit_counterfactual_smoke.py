#!/usr/bin/env python3
"""Audit multi-H option diversity before allowing the full collector."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


OPTIONS = ("base_continue", "detour_complete", "retreat_hold")


def summarize_smoke(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    glass = [dict(row) for row in rows if row.get("condition") == "glass"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in glass:
        grouped[str(row["decision_id"])].append(row)
    contingency = []
    for decision_id, group in grouped.items():
        by_option = {str(row["option"]): str(row["outcome"]) for row in group}
        if set(by_option) != set(OPTIONS) or len(group) != len(OPTIONS):
            raise ValueError(f"decision {decision_id} does not have exactly three options")
        first = group[0]
        contingency.append({
            "decision_id": decision_id,
            "placement_key": first["placement_key"],
            "source_state_sha256": first["source_state_sha256"],
            "horizon_actions": int(first["horizon_actions"]),
            **by_option,
        })
    contingency.sort(key=lambda row: (
        str(row["placement_key"]), -int(row["horizon_actions"])
    ))
    patterns = Counter(tuple(row[option] for option in OPTIONS) for row in contingency)
    detour_advantage = [
        row for row in contingency
        if row["base_continue"] == "catastrophe"
        and row["detour_complete"] == "task_success"
    ]
    retreat_saves = [
        row for row in contingency
        if row["base_continue"] == "catastrophe"
        and row["retreat_hold"] == "safe_noncompletion"
    ]
    distinct_horizons = sorted({row["horizon_actions"] for row in contingency})
    diagnostics = {
        "at_least_three_glass_horizons": len(distinct_horizons) >= 3,
        "at_least_two_option_ordering_patterns": len(patterns) >= 2,
        "smoke_base_catastrophe_detour_success_exists": bool(detour_advantage),
        "base_catastrophe_retreat_safe_exists": bool(retreat_saves),
    }
    return {
        "diagnostics": diagnostics,
        "glass_decision_states": len(contingency),
        "glass_horizons": distinct_horizons,
        "option_ordering_patterns": [
            {"outcomes": dict(zip(OPTIONS, pattern)), "states": count}
            for pattern, count in sorted(patterns.items())
        ],
        "detour_advantage_states": len(detour_advantage),
        "retreat_save_states": len(retreat_saves),
        "contingency": contingency,
    }


def audit(
    smoke_dir: Path,
    config_path: Path,
    sweep_dir: Path,
    expected_commit: str | None,
) -> dict[str, Any]:
    manifest = json.loads((smoke_dir / "capture_manifest.json").read_text())
    config = json.loads(config_path.read_text())
    recorded = manifest["protocol"]["detour"].get("frozen_config")
    if recorded != config:
        raise ValueError("smoke protocol does not exactly match frozen detour config")
    smoke_commit = str(manifest["repository"]["git_commit"])
    if expected_commit is not None and smoke_commit != expected_commit:
        raise ValueError(
            f"smoke commit {smoke_commit} does not match full commit {expected_commit}"
        )
    sweep = json.loads((sweep_dir / "sweep_manifest.json").read_text())
    recommendation = sweep["summary"].get("recommended_config")
    stable_sweep_rescue = bool(
        sweep["summary"].get("go") is True
        and recommendation is not None
        and recommendation.get("config") == config
        and int(recommendation.get("task_success_states", 0)) >= 1
        and not recommendation.get("unstable_states")
    )
    rows = [
        json.loads(line)
        for line in (smoke_dir / "option_rollouts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    smoke_summary = summarize_smoke(rows)
    hard_gate = {
        "frozen_common_config_has_stable_exact_state_rescue": stable_sweep_rescue,
        "smoke_contains_glass_decision_state": smoke_summary["glass_decision_states"] >= 1,
    }
    return {
        "schema_version": 1,
        "kind": "counterfactual_multi_h_smoke_audit",
        "smoke_dir": str(smoke_dir),
        "smoke_commit": smoke_commit,
        "sweep_dir": str(sweep_dir),
        "frozen_detour_config": config,
        "go": all(hard_gate.values()),
        "decision": (
            "minimum paper premise holds; full collection is allowed"
            if all(hard_gate.values())
            else "minimum exact-state rescue premise is missing; do not submit full"
        ),
        "hard_gate": hard_gate,
        **smoke_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", required=True, type=Path)
    parser.add_argument("--detour-config", required=True, type=Path)
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.smoke_dir.resolve(), args.detour_config.resolve(),
        args.sweep_dir.resolve(), args.expected_commit
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload)
    print(payload, end="")
    if not result["go"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
