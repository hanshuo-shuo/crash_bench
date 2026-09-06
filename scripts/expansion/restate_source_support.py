#!/usr/bin/env python3
"""Append-only retrospective support erratum; never authorizes a new test."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.expansion.analyze_benchmark_test import support_counts
from scripts.expansion.train_option_value import read_jsonl, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evidence = {"kind": "retrospective_source_support_erratum", "definition": "contains_label_v2_nonexclusive",
        "confirmatory": False, "test_authorized": False, "historical_results_modified": False,
        "D2": {}, "D5_D8": {}}
    expansion = ROOT / "results/expansion"
    for family, module_name, func in [
        ("fragile", "analyze_fragile_screen", "analyze_fragile_shards"),
        ("staleness", "analyze_staleness_screen", "analyze_staleness_shards"),
        ("action_drift", "analyze_action_drift_screen", "analyze_action_drift_shards"),
        ("narrow", "analyze_narrow_clearance_screen", "analyze_narrow_shards")]:
        files = sorted((expansion / f"d2_{family}_screen").glob("*/task*_source*.json"))
        analyzer = getattr(importlib.import_module("scripts.expansion." + module_name), func)
        result = analyzer([json.loads(p.read_text()) for p in files])
        original_path = next((expansion / f"d2_{family}_screen").glob("*/*screen_analysis.json"))
        original = json.loads(original_path.read_text())
        evidence["D2"][family] = {
            "old_pure_B0_count": original["summary"]["benefit_zero_sources"],
            "corrected_summary": result["summary"], "historical_gate": original["gate"],
            "retrospective_gate_recalculation": result["gate"],
            "inputs_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in files + [original_path]}}
    d5 = next((expansion / "d5_statewise").glob("*/postprocess_*/merged"))
    d8 = next((expansion / "d8_benchmark").glob("*/postprocess_*/merged"))
    for label, directory, roles in [("D5_train_development", d5, {"train", "development"}),
                                    ("D5_development", d5, {"development"}), ("D8_exposed_test", d8, {"confirmatory_id_test"})]:
        anchors = [r for r in read_jsonl(directory / "anchors.jsonl") if r["split_role"] in roles]
        branches = [r for r in read_jsonl(directory / "branches.jsonl") if r["split_role"] in roles]
        if not anchors or not branches:
            raise ValueError(f"empty selected role cohort: {label} {roles}")
        evidence["D5_D8"][label] = {"support": support_counts(anchors, branches),
            "inputs_sha256": {str((directory / f).relative_to(ROOT)): sha256_file(directory / f)
                              for f in ("anchors.jsonl", "branches.jsonl")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
