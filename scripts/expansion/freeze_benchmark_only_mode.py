#!/usr/bin/env python3
"""Freeze the D8-B benchmark-only fallback before any test outcome is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze(
    gate_a: dict[str, Any], gate_b: dict[str, Any], run_manifest: dict[str, Any],
    split_manifest: dict[str, Any], *, hashes: dict[str, str]
) -> dict[str, Any]:
    if gate_a.get("test_rows_read") != 0 or gate_b.get("test_rows_read") != 0:
        raise ValueError("benchmark fallback must freeze before test access")
    if run_manifest.get("test_rows_read") != 0 or run_manifest.get("status") != "SEALED":
        raise ValueError("D6/D7 run is not a sealed zero-test-access artifact")
    if gate_a["gate"]["status"] not in {"GO", "SCOPED_CONTINUE"}:
        raise ValueError("benchmark fallback requires continuing Gate A evidence")
    if gate_b["gate"]["status"] == "GO":
        raise ValueError("benchmark-only mode cannot replace a passing method Gate B")
    if gate_b["gate"]["hard_failures"]:
        raise ValueError("hard model/calibration validity failure blocks any downstream freeze")
    test_sources = sorted(
        (
            {
                key: row[key]
                for key in (
                    "physical_source_id", "mechanism_source_id", "policy_source_id",
                    "mechanism_id", "task_id", "policy_id", "formal_attempt_id",
                    "source_state_sha256", "role",
                )
            }
            for row in split_manifest["assignments"]
            if row["role"] == "confirmatory_id_test"
        ),
        key=lambda row: row["policy_source_id"],
    )
    if len(test_sources) != 32 or len({row["physical_source_id"] for row in test_sources}) != 32:
        raise ValueError("scoped benchmark fallback requires exactly 32 frozen physical test sources")
    task_counts = {
        task: sum(row["task_id"] == task for row in test_sources)
        for task in sorted({row["task_id"] for row in test_sources})
    }
    if task_counts != {"libero_spatial:0": 16, "libero_spatial:2": 16}:
        raise ValueError("scoped benchmark test must retain 16 sources per frozen task")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_benchmark_only_mode_freeze",
        "mode": "SCOPED_BENCHMARK_ONLY__NO_METHOD_SUPERIORITY_TEST",
        "scope": "single_primary_policy_single_staleness_mechanism",
        "reason": "Development Gate B failed claim criteria without hard validity failure.",
        "method_claim_authorized": False,
        "statewise_method_test_authorized": False,
        "benchmark_outcome_collection_authorized": False,
        "test_opened": False,
        "test_outcomes_read": 0,
        "planned_test_physical_sources": 32,
        "task_counts": task_counts,
        "test_source_manifest_sha256": canonical_sha(test_sources),
        "test_sources": test_sources,
        "protocol_sha256": split_manifest["protocol_sha256"],
        **hashes,
        "gate_a_status": gate_a["gate"]["status"],
        "gate_b_status": gate_b["gate"]["status"],
        "gate_b_claim_failures": gate_b["gate"]["claim_scope_failures"],
        "next_action": "FREEZE_D8_B_ANALYSIS_SCRIPT_AND_CREATE_ONE_TIME_TEST_AUTHORIZATION",
        "frozen": True,
    }
    payload["freeze_sha256"] = canonical_sha(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-a", type=Path, required=True)
    parser.add_argument("--gate-b", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark-only freeze: {args.output}")
    paths = {
        "gate_a_artifact_sha256": args.gate_a,
        "gate_b_artifact_sha256": args.gate_b,
        "d6_run_manifest_artifact_sha256": args.run_manifest,
        "split_manifest_artifact_sha256": args.split_manifest,
    }
    payload = freeze(
        json.loads(args.gate_a.read_text()),
        json.loads(args.gate_b.read_text()),
        json.loads(args.run_manifest.read_text()),
        json.loads(args.split_manifest.read_text()),
        hashes={key: sha256_file(path) for key, path in paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "mode": payload["mode"], "sources": 32}, sort_keys=True))


if __name__ == "__main__":
    main()
