#!/usr/bin/env python3
"""Evaluate scoped single-mechanism Gate A from the merged D5 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.utility import option_decision
from crashbench.governance.gates import Criterion, CriterionClass, GatePolicy, evaluate_gate


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze_gate_a(
    anchors: Iterable[dict[str, Any]],
    branches: Iterable[dict[str, Any]],
    invalid: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    all_anchors, all_branches, all_invalid = list(anchors), list(branches), list(invalid)
    allowed_roles = {"train", "development"}
    anchors = [row for row in all_anchors if row.get("split_role") in allowed_roles]
    branches = [row for row in all_branches if row.get("split_role") in allowed_roles]
    invalid = [row for row in all_invalid if row.get("split_role") in allowed_roles]
    calibration_sources_excluded = len(
        {row["physical_source_id"] for row in all_anchors if row.get("split_role") == "calibration"}
    )
    by_block = defaultdict(list)
    for row in branches:
        by_block[row["block_id"]].append(row)
    block_decisions = []
    for anchor in anchors:
        rows = by_block.get(anchor["block_id"], [])
        utilities = {row["option_id"]: float(row["u0"]) for row in rows}
        if set(utilities) != {"base_continue", "observation_refresh", "safe_stop"}:
            continue
        decision = option_decision(
            utilities,
            base_option_id="base_continue",
            deployable_option_ids=utilities,
            gamma=0.10,
        )
        base = next(row for row in rows if row["option_id"] == "base_continue")
        block_decisions.append(
            {
                "block_id": anchor["block_id"],
                "physical_source_id": anchor["physical_source_id"],
                "task_id": anchor["task_id"],
                "split_role": anchor["split_role"],
                "condition": anchor["condition"],
                "severity_id": anchor["severity_id"],
                "anchor_steps": anchor["anchor_steps"],
                "base_catastrophe": int(base["outcome"]["catastrophe"]),
                "benefit": decision.benefit,
                "strict_winner": decision.strict_winner,
                "best_advantage": decision.best_deployable_advantage,
            }
        )
    by_source = defaultdict(list)
    for row in block_decisions:
        by_source[row["physical_source_id"]].append(row)
    source_rows = []
    strict_support = defaultdict(set)
    same_risk_flip_sources = set()
    for source, rows in sorted(by_source.items()):
        winners = {row["strict_winner"] for row in rows if row["strict_winner"]}
        for winner in winners:
            strict_support[winner].add(source)
        has_flip = False
        for risk in (0, 1):
            matched = [row for row in rows if row["base_catastrophe"] == risk]
            signatures = {(row["benefit"], row["strict_winner"]) for row in matched}
            if len(signatures) >= 2:
                has_flip = True
        if has_flip:
            same_risk_flip_sources.add(source)
        source_rows.append(
            {
                "physical_source_id": source,
                "task_id": rows[0]["task_id"],
                "split_role": rows[0]["split_role"],
                "complete_blocks": len(rows),
                "benefit": any(row["benefit"] for row in rows),
                "strict_winners": sorted(winners),
                "same_risk_flip": has_flip,
            }
        )
    task_invalid_rates = {}
    for task in ("libero_spatial:0", "libero_spatial:2"):
        accepted_count = sum(row["task_id"] == task for row in anchors)
        invalid_count = sum(row["task_id"] == task for row in invalid)
        denominator = accepted_count + invalid_count
        task_invalid_rates[task] = invalid_count / denominator if denominator else 1.0
    benefit_one = sum(row["benefit"] for row in source_rows)
    benefit_zero = len(source_rows) - benefit_one
    exact_rate = sum(row.get("exact_branch_start") is True for row in anchors) / len(anchors) if anchors else 0
    hard = CriterionClass.HARD_VALIDITY
    claim = CriterionClass.CLAIM_SCOPE
    criteria = [
        Criterion("source_shard_count", hard, manifest.get("physical_source_count"), "==", 48),
        Criterion("train_development_source_count", hard, len(source_rows), "==", 36),
        Criterion("merge_status", hard, manifest.get("status"), "==", "GO"),
        Criterion("test_rows_read", hard, manifest.get("test_rows_read"), "==", 0),
        Criterion("exact_branch_start_rate", hard, exact_rate, "==", 1.0),
        Criterion("task0_invalid_rate", hard, task_invalid_rates["libero_spatial:0"], "<=", 0.10),
        Criterion("task2_invalid_rate", hard, task_invalid_rates["libero_spatial:2"], "<=", 0.10),
        Criterion("benefit_zero_sources", claim, benefit_zero, ">=", 8),
        Criterion("benefit_one_sources", claim, benefit_one, ">=", 8),
        Criterion(
            "strict_observation_refresh_sources", claim,
            len(strict_support["observation_refresh"]), ">=", 6
        ),
        Criterion("strict_safe_stop_sources", claim, len(strict_support["safe_stop"]), ">=", 6),
        Criterion("same_risk_flip_sources", claim, len(same_risk_flip_sources), ">=", 4),
    ]
    gate = evaluate_gate(
        criteria,
        GatePolicy(
            stage_id="D5_SCOPED_STALENESS_GATE_A",
            confirmatory=False,
            test_outcomes_opened=False,
            allow_scoped_continuation=True,
            go_next_action="TRAIN_ODUR_AND_FROZEN_BASELINES",
            scoped_next_action="CONTINUE_METHOD_PILOT_WITH_LIMITED_SUPPORT_CLAIMS",
            fail_next_action="STOP_MODEL_TRAINING_AND_AUDIT_D5_VALIDITY",
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": "crashbench_expansion_d5_gate_a_analysis",
        "gate": gate,
        "source_count": len(source_rows),
        "roles_analyzed": ["development", "train"],
        "calibration_sources_excluded": calibration_sources_excluded,
        "calibration_rows_read": 0,
        "benefit_zero_sources": benefit_zero,
        "benefit_one_sources": benefit_one,
        "strict_support_sources": {
            option: len(sources) for option, sources in sorted(strict_support.items())
        },
        "same_risk_flip_sources": len(same_risk_flip_sources),
        "task_invalid_rates": task_invalid_rates,
        "exact_branch_start_rate": exact_rate,
        "sources": source_rows,
        "block_decision_count": len(block_decisions),
        "test_rows_read": 0,
    }
    payload["analysis_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Gate A analysis: {args.output}")
    payload = analyze_gate_a(
        read_jsonl(args.merged_dir / "anchors.jsonl"),
        read_jsonl(args.merged_dir / "branches.jsonl"),
        read_jsonl(args.merged_dir / "invalid_blocks.jsonl"),
        json.loads((args.merged_dir / "dataset_manifest.json").read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["gate"]["status"], "next_action": payload["gate"]["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
