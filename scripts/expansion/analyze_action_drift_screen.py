#!/usr/bin/env python3
"""Source-aware D2 screen decision for action drift v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.support import source_label_support, aggregate_source_support
from crashbench.data.utility import option_decision
from crashbench.mechanisms.registry import evaluate_mechanism_screen


EXPECTED_OPTIONS = {"base_continue", "corrective_requery", "safe_stop"}


def _block(raw: dict[str, Any]) -> dict[str, Any]:
    outcomes = raw.get("option_outcomes", [])
    utilities = {row["option_id"]: float(row["utility"]) for row in outcomes}
    exact = (
        len(outcomes) == 3
        and len({row["branch_start_bundle_id"] for row in outcomes}) == 1
        and len({row["injector_state_sha256"] for row in outcomes}) == 1
    )
    if set(utilities) != EXPECTED_OPTIONS:
        return {"complete": False, "exact_start": exact}
    decision = option_decision(
        utilities,
        base_option_id="base_continue",
        deployable_option_ids=utilities,
        gamma=0.10,
    )
    base = next(row for row in outcomes if row["option_id"] == "base_continue")
    return {
        "complete": True,
        "exact_start": exact,
        "benefit": decision.benefit,
        "strict_winner": decision.strict_winner,
        "base_catastrophe": bool(base["outcome"]["catastrophe"]),
        "base_success": bool(base["outcome"]["task_success"]),
    }


def analyze_action_drift_shards(shards: Iterable[dict[str, Any]]) -> dict[str, Any]:
    shards = list(shards)
    keys = [(int(row.get("task_id", -1)), int(row.get("source_index", -1))) for row in shards]
    expected = {(task, source) for task in (0, 2) for source in range(8)}
    duplicate = sorted({key for key in keys if keys.count(key) > 1})
    missing = sorted(expected - set(keys))
    unexpected = sorted(set(keys) - expected)
    sources, blocks = [], []
    for shard in sorted(shards, key=lambda row: (int(row["task_id"]), int(row["source_index"]))):
        complete, winners = [], set()
        for raw in shard.get("blocks", []):
            if raw.get("status") != "COMPLETE_REALIZED_OPTIONS":
                continue
            summary = _block(raw)
            summary.update(
                {
                    "condition": raw["condition"],
                    "severity_id": raw["severity_id"],
                    "task_id": int(shard["task_id"]),
                    "source_index": int(shard["source_index"]),
                }
            )
            blocks.append(summary)
            if summary["complete"]:
                complete.append(summary)
                if summary["strict_winner"] not in {None, "base_continue"}:
                    winners.add(summary["strict_winner"])
        sources.append(
            {
                "task_id": int(shard["task_id"]),
                "source_index": int(shard["source_index"]),
                "eligible": bool(complete),
                "complete_blocks": len(complete),
                "benefit": any(row["benefit"] for row in complete),
                **source_label_support(row["benefit"] for row in complete),
                "strict_intervention_winners": sorted(winners),
                "two_distinct_strict_interventions": len(winners) >= 2,
            }
        )
    eligible = [row for row in sources if row["eligible"]]
    complete_blocks = [row for row in blocks if row.get("complete")]
    drift = [row for row in complete_blocks if row["condition"] == "drift"]
    controls = [
        row
        for row in complete_blocks
        if row["condition"] in {"zero_bias_control", "orthogonal_bias_control"}
    ]
    drift_cat = sum(row["base_catastrophe"] for row in drift) / len(drift) if drift else 0
    control_cat = sum(row["base_catastrophe"] for row in controls) / len(controls) if controls else 1
    drift_success = sum(row["base_success"] for row in drift) / len(drift) if drift else 0
    control_success = sum(row["base_success"] for row in controls) / len(controls) if controls else 0
    summary = {
        "task0_nominal_success": 0.875,
        "task2_nominal_success": 1.0,
        "task0_eligible_sources": sum(row["eligible"] and row["task_id"] == 0 for row in sources),
        "task2_eligible_sources": sum(row["eligible"] and row["task_id"] == 2 for row in sources),
        "exact_restore_rate": sum(row["exact_start"] for row in complete_blocks) / len(complete_blocks) if complete_blocks else 0,
        "admissible_execution_rate": len(complete_blocks) / len(blocks) if blocks else 0,
        "matched_control_catastrophe": control_cat,
        "mechanism_attribution_valid": bool(drift and controls)
        and (drift_cat > control_cat or drift_success < control_success),
        **aggregate_source_support(eligible),
        "task0_benefit_sources": sum(row["benefit"] and row["task_id"] == 0 for row in eligible),
        "task2_benefit_sources": sum(row["benefit"] and row["task_id"] == 2 for row in eligible),
        "two_distinct_strict_winner_sources": sum(
            row["two_distinct_strict_interventions"] for row in eligible
        ),
        "drift_base_catastrophe": drift_cat,
        "drift_base_success": drift_success,
        "control_base_success": control_success,
    }
    gate = evaluate_mechanism_screen(mechanism_id="action_drift_v1", summary=summary)
    if duplicate or missing or unexpected:
        gate = {**gate, "status": "NO_GO", "next_action": "RESOLVE_ACTION_DRIFT_SHARD_ACCOUNTING"}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_action_drift_screen_analysis",
        "mechanism_id": "action_drift_v1",
        "observed_sources": len(shards),
        "duplicate_keys": [list(key) for key in duplicate],
        "missing_keys": [list(key) for key in missing],
        "unexpected_keys": [list(key) for key in unexpected],
        "summary": summary,
        "sources": sources,
        "complete_block_count": len(complete_blocks),
        "gate": gate,
    }
    payload["analysis_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite action-drift analysis: {args.output}")
    payload = analyze_action_drift_shards(json.loads(path.read_text()) for path in args.shard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["gate"]["status"], "next_action": payload["gate"]["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
