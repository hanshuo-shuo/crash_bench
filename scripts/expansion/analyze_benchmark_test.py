#!/usr/bin/env python3
"""Confirmatory D8-B benchmark-only heterogeneity analysis on 32 frozen sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.utility import UtilityWeights, option_decision, scalar_utility
from crashbench.governance.gates import Criterion, CriterionClass, GatePolicy, evaluate_gate
from scripts.expansion.merge_statewise_dataset import budgets_from_config, outcome_vector


OPTIONS = {"base_continue", "observation_refresh", "safe_stop"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def support_counts(
    anchors: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    *,
    utility_override: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in branches:
        by_block[str(row["block_id"])].append(row)
    decisions = []
    for anchor in anchors:
        block = str(anchor["block_id"])
        rows = by_block.get(block, [])
        utilities = {
            str(row["option_id"]): (
                float(row["u0"])
                if utility_override is None
                else float(utility_override[(block, str(row["option_id"]))])
            )
            for row in rows
        }
        if set(utilities) != OPTIONS:
            continue
        decision = option_decision(
            utilities, base_option_id="base_continue", deployable_option_ids=utilities, gamma=0.10
        )
        base = next(row for row in rows if row["option_id"] == "base_continue")
        decisions.append(
            {
                "physical_source_id": anchor["physical_source_id"],
                "base_catastrophe": int(base["outcome"]["catastrophe"]),
                "benefit": decision.benefit,
                "strict_winner": decision.strict_winner,
            }
        )
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        by_source[str(row["physical_source_id"])].append(row)
    benefit_one = 0
    strict_support: dict[str, set[str]] = defaultdict(set)
    flip_sources = set()
    source_rows = []
    for source, rows in sorted(by_source.items()):
        source_benefit = any(row["benefit"] for row in rows)
        benefit_one += source_benefit
        winners = {str(row["strict_winner"]) for row in rows if row["strict_winner"]}
        for winner in winners:
            strict_support[winner].add(source)
        flip = any(
            len({(row["benefit"], row["strict_winner"]) for row in rows if row["base_catastrophe"] == risk}) >= 2
            for risk in (0, 1)
        )
        if flip:
            flip_sources.add(source)
        source_rows.append(
            {"physical_source_id": source, "benefit": source_benefit, "strict_winners": sorted(winners), "same_risk_flip": flip}
        )
    return {
        "source_count": len(source_rows),
        "benefit_one_sources": benefit_one,
        "benefit_zero_sources": len(source_rows) - benefit_one,
        "strict_support_sources": {option: len(strict_support[option]) for option in sorted(OPTIONS)},
        "same_risk_flip_sources": len(flip_sources),
        "block_decision_count": len(decisions),
        "sources": source_rows,
    }


def analyze(
    anchors: list[dict[str, Any]], branches: list[dict[str, Any]], invalid: list[dict[str, Any]],
    manifest: dict[str, Any], utility_config: dict[str, Any], benchmark_freeze: dict[str, Any]
) -> dict[str, Any]:
    if benchmark_freeze["mode"] != "SCOPED_BENCHMARK_ONLY__NO_METHOD_SUPERIORITY_TEST":
        raise ValueError("D8-B analysis requires benchmark-only freeze")
    primary = support_counts(anchors, branches)
    budgets = budgets_from_config(utility_config)
    sensitivity = []
    for catastrophe_cost in (1.0, 2.0, 5.0):
        weights = UtilityWeights(catastrophe=-catastrophe_cost)
        values = {
            (str(row["block_id"]), str(row["option_id"])): scalar_utility(
                outcome_vector(row["outcome"]), budgets, weights
            )
            for row in branches
        }
        counts = support_counts(anchors, branches, utility_override=values)
        counts["catastrophe_cost"] = catastrophe_cost
        counts["scoped_heterogeneity_supported"] = (
            counts["benefit_zero_sources"] >= 3
            and counts["benefit_one_sources"] >= 8
            and counts["strict_support_sources"]["observation_refresh"] >= 4
            and counts["strict_support_sources"]["safe_stop"] >= 4
            and counts["same_risk_flip_sources"] >= 3
        )
        sensitivity.append(counts)
    task_sources = {
        task: len({row["physical_source_id"] for row in anchors if row["task_id"] == task})
        for task in ("libero_spatial:0", "libero_spatial:2")
    }
    exact_rate = sum(row.get("exact_branch_start") is True for row in anchors) / len(anchors) if anchors else 0.0
    invalid_rate = len(invalid) / (len(invalid) + len(anchors)) if anchors or invalid else 1.0
    hard, claim = CriterionClass.HARD_VALIDITY, CriterionClass.CLAIM_SCOPE
    criteria = [
        Criterion("merged_status", hard, manifest.get("status"), "==", "GO"),
        Criterion("physical_sources", hard, primary["source_count"], "==", 32),
        Criterion("task0_sources", hard, task_sources["libero_spatial:0"], "==", 16),
        Criterion("task2_sources", hard, task_sources["libero_spatial:2"], "==", 16),
        Criterion("test_source_rows_read", hard, manifest.get("test_rows_read"), "==", 32),
        Criterion("exact_branch_start_rate", hard, exact_rate, "==", 1.0),
        Criterion("mechanical_invalid_rate", hard, invalid_rate, "<=", 0.10),
        Criterion("benefit_zero_sources", claim, primary["benefit_zero_sources"], ">=", 3),
        Criterion("benefit_one_sources", claim, primary["benefit_one_sources"], ">=", 8),
        Criterion("strict_refresh_sources", claim, primary["strict_support_sources"]["observation_refresh"], ">=", 4),
        Criterion("strict_safe_stop_sources", claim, primary["strict_support_sources"]["safe_stop"], ">=", 4),
        Criterion("same_risk_flip_sources", claim, primary["same_risk_flip_sources"], ">=", 3),
        Criterion(
            "catastrophe_sensitivity_settings", claim,
            sum(row["scoped_heterogeneity_supported"] for row in sensitivity), ">=", 2,
        ),
    ]
    gate = evaluate_gate(
        criteria,
        GatePolicy(
            stage_id="D8_B_SCOPED_CONFIRMATORY_BENCHMARK_GATE_A_TEST",
            confirmatory=True,
            test_outcomes_opened=True,
            allow_scoped_continuation=False,
            go_next_action="AUTHOR_SCOPED_BENCHMARK_RELEASE",
            scoped_next_action="UNUSED",
            fail_next_action="RELEASE_COMPLETE_TEST_SCOPE_FAILURE_WITHOUT_BROAD_CLAIM",
        ),
    )
    if gate["status"] != "GO":
        gate["status"] = "TEST_SCOPE_FAILURE"
        gate.pop("decision_sha256", None)
        gate["decision_sha256"] = hashlib.sha256(
            json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d8_benchmark_test_analysis",
        "scope": "single_primary_policy_single_staleness_mechanism",
        "prospective_adaptation": (
            "The frozen 32-source single-mechanism fallback uses B0>=3, B1>=8, each strict "
            "intervention>=4, and same-risk flips>=3; cross-mechanism criteria are inapplicable."
        ),
        "gate": gate,
        "primary": primary,
        "task_source_counts": task_sources,
        "exact_branch_start_rate": exact_rate,
        "mechanical_invalid_rate": invalid_rate,
        "utility_sensitivity": sensitivity,
        "test_sources_read": 32,
        "method_superiority_evaluated": False,
    }
    payload["analysis_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--benchmark-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D8-B analysis: {args.output}")
    payload = analyze(
        read_jsonl(args.merged_dir / "anchors.jsonl"),
        read_jsonl(args.merged_dir / "branches.jsonl"),
        read_jsonl(args.merged_dir / "invalid_blocks.jsonl"),
        json.loads((args.merged_dir / "dataset_manifest.json").read_text()),
        json.loads(args.utility_config.read_text()),
        json.loads(args.benchmark_freeze.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["gate"]["status"], "next_action": payload["gate"]["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
