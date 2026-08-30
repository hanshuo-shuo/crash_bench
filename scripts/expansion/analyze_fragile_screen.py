#!/usr/bin/env python3
"""Source-aware D2 screen decision for fragile-path collision v2."""

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

from crashbench.data.utility import option_decision
from crashbench.mechanisms.registry import evaluate_mechanism_screen


def _complete_block_summary(block: dict[str, Any]) -> dict[str, Any]:
    outcomes = block.get("option_outcomes", [])
    utilities = {row["option_id"]: float(row["utility"]) for row in outcomes}
    exact_start = (
        len(outcomes) == 3
        and len({row["branch_start_bundle_id"] for row in outcomes}) == 1
        and len(
            {
                json.dumps(row["branch_start_component_hashes"], sort_keys=True)
                for row in outcomes
            }
        )
        == 1
    )
    if set(utilities) != {"base_continue", "backtrack_requery", "safe_stop"}:
        return {"complete": False, "exact_start": exact_start}
    decision = option_decision(
        utilities,
        base_option_id="base_continue",
        deployable_option_ids=utilities,
        epsilon=0,
        gamma=0.10,
    )
    base = next(row for row in outcomes if row["option_id"] == "base_continue")
    return {
        "complete": True,
        "exact_start": exact_start,
        "benefit": decision.benefit,
        "strict_winner": decision.strict_winner,
        "best_options": list(decision.best_options),
        "best_advantage": decision.best_deployable_advantage,
        "base_catastrophe": bool(base["outcome"]["catastrophe"]),
    }


def analyze_fragile_shards(shards: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(shards)
    keys = [(int(row.get("task_id", -1)), int(row.get("source_index", -1))) for row in rows]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    expected_keys = {(task, source) for task in (0, 2) for source in range(8)}
    missing_keys = sorted(expected_keys - set(keys))
    unexpected_keys = sorted(set(keys) - expected_keys)
    source_rows = []
    block_summaries = []
    for shard in sorted(rows, key=lambda row: (int(row["task_id"]), int(row["source_index"]))):
        complete_blocks = []
        strict_interventions = set()
        for block in shard.get("blocks", []):
            if block.get("status") != "COMPLETE_REALIZED_OPTIONS":
                continue
            summary = _complete_block_summary(block)
            summary.update(
                {
                    "task_id": int(shard["task_id"]),
                    "source_index": int(shard["source_index"]),
                    "condition": block["condition"],
                    "severity_id": block["severity_id"],
                }
            )
            block_summaries.append(summary)
            if summary["complete"]:
                complete_blocks.append(summary)
                if summary["strict_winner"] not in {None, "base_continue"}:
                    strict_interventions.add(summary["strict_winner"])
        eligible = bool(complete_blocks)
        source_rows.append(
            {
                "task_id": int(shard["task_id"]),
                "source_index": int(shard["source_index"]),
                "attempt_id": shard.get("attempt_id"),
                "eligible": eligible,
                "complete_blocks": len(complete_blocks),
                "benefit": any(row["benefit"] for row in complete_blocks),
                "strict_intervention_winners": sorted(strict_interventions),
                "two_distinct_strict_interventions": len(strict_interventions) >= 2,
            }
        )
    eligible_sources = [row for row in source_rows if row["eligible"]]
    control_blocks = [
        row
        for row in block_summaries
        if row.get("complete") and row["condition"] in {"off_path_control", "no_object_control"}
    ]
    onpath_blocks = [
        row for row in block_summaries if row.get("complete") and row["condition"] == "on_path"
    ]
    exact_blocks = [row for row in block_summaries if row.get("complete") and row["exact_start"]]
    summary = {
        "task0_nominal_success": 0.75,
        "task2_nominal_success": 1.0,
        "task0_eligible_sources": sum(row["eligible"] and row["task_id"] == 0 for row in source_rows),
        "task2_eligible_sources": sum(row["eligible"] and row["task_id"] == 2 for row in source_rows),
        "exact_restore_rate": len(exact_blocks) / len(block_summaries) if block_summaries else 0.0,
        "admissible_execution_rate": sum(row.get("complete", False) for row in block_summaries) / len(block_summaries) if block_summaries else 0.0,
        "matched_control_catastrophe": sum(row["base_catastrophe"] for row in control_blocks) / len(control_blocks) if control_blocks else 1.0,
        "mechanism_attribution_valid": bool(onpath_blocks)
        and any(row["base_catastrophe"] for row in onpath_blocks)
        and bool(control_blocks),
        "benefit_zero_sources": sum(not row["benefit"] for row in eligible_sources),
        "benefit_one_sources": sum(row["benefit"] for row in eligible_sources),
        "task0_benefit_sources": sum(row["benefit"] and row["task_id"] == 0 for row in eligible_sources),
        "task2_benefit_sources": sum(row["benefit"] and row["task_id"] == 2 for row in eligible_sources),
        "two_distinct_strict_winner_sources": sum(
            row["two_distinct_strict_interventions"] for row in eligible_sources
        ),
    }
    gate = evaluate_mechanism_screen(
        mechanism_id="fragile_path_collision_v2", summary=summary
    )
    if duplicate_keys or missing_keys or unexpected_keys:
        gate = {
            **gate,
            "status": "NO_GO",
            "next_action": "RESOLVE_FRAGILE_SHARD_IDENTITY_OR_ACCOUNTING",
        }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_fragile_screen_analysis",
        "mechanism_id": "fragile_path_collision_v2",
        "expected_sources": 16,
        "observed_sources": len(rows),
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "missing_keys": [list(key) for key in missing_keys],
        "unexpected_keys": [list(key) for key in unexpected_keys],
        "summary": summary,
        "sources": source_rows,
        "complete_block_count": sum(row.get("complete", False) for row in block_summaries),
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
        raise FileExistsError(f"refusing to overwrite fragile analysis: {args.output}")
    payload = analyze_fragile_shards(json.loads(path.read_text()) for path in args.shard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["gate"]["status"], "next_action": payload["gate"]["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
