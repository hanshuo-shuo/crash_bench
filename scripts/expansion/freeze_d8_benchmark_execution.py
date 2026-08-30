#!/usr/bin/env python3
"""Freeze D8-B code identities and atomically open its one-time test lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.splits import TestAuthorization, create_test_authorization
from scripts.expansion.hash_tree_manifest import resolve_git_head


PINNED_EXECUTION_FILES = (
    "scripts/expansion/collect_staleness_statewise.py",
    "scripts/expansion/audit_benchmark_test_shards.py",
    "scripts/expansion/merge_benchmark_test_dataset.py",
    "scripts/expansion/analyze_benchmark_test.py",
    "setup/expansion_benchmark_test.sbatch",
    "setup/submit_expansion_benchmark_test.sh",
    "configs/expansion/utility_v1.yaml",
    "configs/expansion/deployable_options_v1.yaml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-freeze", type=Path, required=True)
    parser.add_argument("--d6-run-manifest", type=Path, required=True)
    parser.add_argument("--baseline-selection", type=Path, required=True)
    parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite D8-B execution root: {args.output_dir}")
    benchmark = json.loads(args.benchmark_freeze.read_text())
    if benchmark["next_action"] != "FREEZE_D8_B_ANALYSIS_SCRIPT_AND_CREATE_ONE_TIME_TEST_AUTHORIZATION":
        raise ValueError("benchmark freeze does not authorize D8-B execution freeze")
    if benchmark["test_opened"] is not False or benchmark["test_outcomes_read"] != 0:
        raise ValueError("D8-B execution must freeze before test access")
    files = [
        {"path": relative, "sha256": sha256_file(ROOT / relative)}
        for relative in PINNED_EXECUTION_FILES
    ]
    bundle_sha = canonical_sha(files)
    authorization = TestAuthorization(
        run_id=args.run_id,
        protocol_sha256=benchmark["protocol_sha256"],
        test_source_manifest_sha256=benchmark["test_source_manifest_sha256"],
        model_sha256=sha256_file(args.d6_run_manifest),
        comparator_sha256=sha256_file(args.baseline_selection),
        calibration_sha256=sha256_file(args.calibration_freeze),
        analysis_script_sha256=bundle_sha,
    )
    args.output_dir.mkdir(parents=True)
    auth_payload = create_test_authorization(args.output_dir / "authorization", authorization)
    payload = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d8_benchmark_execution_freeze",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolve_git_head(ROOT),
        "benchmark_freeze_sha256": sha256_file(args.benchmark_freeze),
        "execution_files": files,
        "analysis_script_sha256": bundle_sha,
        "authorization_token": auth_payload["authorization_token"],
        "benchmark_outcome_collection_authorized": True,
        "method_superiority_test_authorized": False,
        "planned_array": "0-31",
        "planned_test_sources": 32,
        "test_outcomes_read": 0,
        "frozen": True,
    }
    payload["freeze_sha256"] = canonical_sha(payload)
    output = args.output_dir / "execution_freeze.json"
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(args.output_dir), "run_id": args.run_id, "authorization_token": auth_payload["authorization_token"]}, sort_keys=True))


if __name__ == "__main__":
    main()
