#!/usr/bin/env python3
"""Supplemental non-gating analysis of all 108 predeclared utility weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.utility import scalar_utility, sensitivity_weights
from scripts.expansion.analyze_benchmark_test import read_jsonl, support_counts
from scripts.expansion.merge_statewise_dataset import budgets_from_config, outcome_vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite supplemental utility sensitivity: {args.output}")
    anchors = read_jsonl(args.merged_dir / "anchors.jsonl")
    branches = read_jsonl(args.merged_dir / "branches.jsonl")
    manifest = json.loads((args.merged_dir / "dataset_manifest.json").read_text())
    if manifest.get("kind") != "crashbench_expansion_d8_benchmark_test_merged_manifest":
        raise ValueError("full sensitivity requires the sealed D8-B merged dataset")
    budgets = budgets_from_config(json.loads(args.utility_config.read_text()))
    rows = []
    for index, weights in enumerate(sensitivity_weights()):
        values = {
            (str(row["block_id"]), str(row["option_id"])): scalar_utility(
                outcome_vector(row["outcome"]), budgets, weights
            )
            for row in branches
        }
        support = support_counts(anchors, branches, utility_override=values)
        by_source_option: dict[tuple[str, str], list[float]] = defaultdict(list)
        for branch in branches:
            by_source_option[(str(branch["physical_source_id"]), str(branch["option_id"]))].append(
                values[(str(branch["block_id"]), str(branch["option_id"]))]
            )
        option_values = {
            option: float(np.mean([
                np.mean(values_for_source)
                for (source, current_option), values_for_source in by_source_option.items()
                if current_option == option
            ]))
            for option in sorted({row["option_id"] for row in branches})
        }
        winner = max(sorted(option_values), key=lambda option: option_values[option])
        rows.append(
            {
                "weight_id": f"utility_grid_{index:03d}",
                "weights": asdict(weights),
                "best_fixed_option": winner,
                "best_fixed_source_macro_u": option_values[winner],
                "option_source_macro_u": option_values,
                "benefit_zero_sources": support["benefit_zero_sources"],
                "benefit_one_sources": support["benefit_one_sources"],
                "strict_support_sources": support["strict_support_sources"],
                "same_risk_flip_sources": support["same_risk_flip_sources"],
            }
        )
    if len(rows) != 108:
        raise RuntimeError("predeclared utility grid no longer has 108 settings")
    payload = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d8_full_utility_sensitivity",
        "analysis_role": "SUPPLEMENTAL_NON_GATING_PREDECLARED_WEIGHT_GRID",
        "confirmatory_gate_changed": False,
        "method_ranking_claim": False,
        "weight_setting_count": len(rows),
        "test_sources_read": manifest["test_rows_read"],
        "rows": rows,
    }
    payload["analysis_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "weights": 108, "confirmatory_gate_changed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
