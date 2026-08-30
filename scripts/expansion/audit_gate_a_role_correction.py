#!/usr/bin/env python3
"""Audit the post-open D5 Gate-A role-filter correction without changing D8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-gate-a", type=Path, required=True)
    parser.add_argument("--corrected-gate-a", type=Path, required=True)
    parser.add_argument("--benchmark-freeze", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Gate-A correction audit: {args.output}")
    original = json.loads(args.original_gate_a.read_text())
    corrected = json.loads(args.corrected_gate_a.read_text())
    benchmark = json.loads(args.benchmark_freeze.read_text())
    execution = json.loads(args.execution_freeze.read_text())
    authorization = json.loads(args.authorization.read_text())
    if original["source_count"] != 48 or corrected["source_count"] != 36:
        raise ValueError("Gate-A correction does not have the expected 48-to-36 role filter")
    if corrected["calibration_sources_excluded"] != 12 or corrected["calibration_rows_read"] != 0:
        raise ValueError("corrected Gate A did not fully exclude calibration")
    if original["gate"]["status"] != corrected["gate"]["status"]:
        raise ValueError("Gate-A role correction changes downstream stage status")
    if original["gate"]["next_action"] != corrected["gate"]["next_action"]:
        raise ValueError("Gate-A role correction changes downstream next action")
    if authorization["test_source_manifest_sha256"] != benchmark["test_source_manifest_sha256"]:
        raise ValueError("D8 authorization source identities drift from benchmark freeze")
    if authorization["analysis_script_sha256"] != execution["analysis_script_sha256"]:
        raise ValueError("D8 authorization analysis identity drift")
    pinned_paths = {row["path"] for row in execution["execution_files"]}
    if "scripts/expansion/analyze_statewise_gate_a.py" in pinned_paths:
        raise ValueError("D8 execution unexpectedly pinned the corrected development analyzer")
    payload = {
        "schema_version": 1,
        "kind": "crashbench_expansion_gate_a_role_filter_correction_audit",
        "discovered_after_test_open": True,
        "slurm_job_id": args.slurm_job_id,
        "original_gate_a_sha256": sha256_file(args.original_gate_a),
        "corrected_gate_a_sha256": sha256_file(args.corrected_gate_a),
        "original_counts": {
            "sources": original["source_count"],
            "benefit_zero": original["benefit_zero_sources"],
            "benefit_one": original["benefit_one_sources"],
            "same_risk_flip": original["same_risk_flip_sources"],
        },
        "corrected_train_development_counts": {
            "sources": corrected["source_count"],
            "benefit_zero": corrected["benefit_zero_sources"],
            "benefit_one": corrected["benefit_one_sources"],
            "same_risk_flip": corrected["same_risk_flip_sources"],
            "strict_support": corrected["strict_support_sources"],
        },
        "status_unchanged": True,
        "next_action_unchanged": True,
        "test_source_manifest_unchanged": True,
        "d8_thresholds_changed": False,
        "d8_execution_files_changed": False,
        "d8_outcomes_inspected_for_correction": False,
        "continue_same_authorized_d8_run": True,
        "reauthorization_or_reopen_permitted": False,
        "release_requirement": (
            "Supersede the original D5 Gate-A support counts with the corrected train+development-only artifact; "
            "report that the correction was discovered after D8 opened and did not alter D8 identities or gates."
        ),
    }
    payload["audit_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "continue_same_run": True, "status": corrected["gate"]["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
