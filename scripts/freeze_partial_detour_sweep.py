#!/usr/bin/env python3
"""Freeze a common detour config from an auditable partial sweep checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_counterfactual_detour import build_configs


def select_stable_success(
    rows: Iterable[Mapping[str, Any]],
    configs: Iterable[Mapping[str, Any]],
    *,
    min_replicates: int,
) -> dict[str, Any]:
    configs_by_id = {str(row["config_id"]): dict(row) for row in configs}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        grouped[(str(row["config_id"]), str(row["state_id"]))].append(row)
    candidates = []
    for (config_id, state_id), group in grouped.items():
        if len(group) < min_replicates:
            continue
        if {str(row["outcome"]) for row in group} != {"task_success"}:
            continue
        candidates.append({
            "config_id": config_id,
            "config": {
                key: value for key, value in configs_by_id[config_id].items()
                if key != "config_id"
            },
            "state_id": state_id,
            "replicates": len(group),
            "task_success_states": 1,
            "unstable_states": [],
            "mean_peak_force_n": sum(float(row["peak_force_n"]) for row in group) / len(group),
            "mean_steps": sum(int(row["steps"]) for row in group) / len(group),
        })
    candidates.sort(key=lambda row: (
        float(row["mean_peak_force_n"]), float(row["mean_steps"]), str(row["config_id"])
    ))
    if not candidates:
        raise ValueError("partial sweep has no replicated stable task-success cell")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-replicates", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    rows = [
        json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()
    ]
    configs = build_configs(
        sides=[-1.0, 1.0], lane_margins=[0.12, 0.18],
        lift_offsets=[0.30, 0.38], descend_offsets=[0.018, 0.04],
        grasp_xy_offsets=[[0.009, -0.04], [-0.003, -0.05], [0.013, -0.06]],
        departure_clearance=0.06,
    )
    winner = select_stable_success(rows, configs, min_replicates=args.min_replicates)
    args.output.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "kind": "counterfactual_detour_partial_checkpoint_freeze",
        "partial_checkpoint": True,
        "rows_observed": len(rows),
        "minimum_replicates": args.min_replicates,
        "summary": {
            "go": True,
            "decision": "freeze lowest-force replicated success from partial checkpoint",
            "recommended_config": winner,
        },
    }
    (args.output / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "frozen_detour_config.json").write_text(
        json.dumps(winner["config"], indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
