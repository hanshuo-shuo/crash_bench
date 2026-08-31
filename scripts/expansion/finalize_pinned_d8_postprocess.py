#!/usr/bin/env python3
"""Finalize provenance for an already sealed pinned D8-B postprocess tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.expansion.hash_tree_manifest import resolve_git_head


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--postprocess-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--collection-job-id", required=True)
    parser.add_argument("--postprocess-job-id", required=True)
    args = parser.parse_args()
    output = args.postprocess_dir / "run_manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite D8 postprocess manifest: {output}")
    audit_path = args.postprocess_dir / "raw_shard_audit.json"
    merge_path = args.postprocess_dir / "merged" / "dataset_manifest.json"
    analysis_path = args.postprocess_dir / "benchmark_test_analysis.json"
    seal_path = args.run_root / "authorization" / "test_complete.seal"
    audit = json.loads(audit_path.read_text())
    merged = json.loads(merge_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    seal = json.loads(seal_path.read_text())
    if audit["status"] != "GO" or audit["observed_shards"] != 32:
        raise ValueError("D8 raw audit is not complete")
    if merged["status"] != "GO" or merged["physical_source_count"] != 32:
        raise ValueError("D8 merged dataset is not complete")
    if seal["completeness_audit_sha256"] != audit["audit_sha256"]:
        raise ValueError("D8 completion seal does not pin the raw audit")
    payload = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d8_benchmark_postprocess_manifest",
        "execution_git_commit": args.execution_commit,
        "postprocess_manifest_git_commit": resolve_git_head(Path.cwd()),
        "collection_slurm_job_id": args.collection_job_id,
        "postprocess_slurm_job_id": args.postprocess_job_id,
        "physical_source_shards": 32,
        "anchor_count": merged["anchor_count"],
        "branch_count": merged["branch_count"],
        "invalid_block_count": merged["invalid_block_count"],
        "test_sources_read": merged["test_rows_read"],
        "method_superiority_evaluated": False,
        "raw_audit_sha256": audit["audit_sha256"],
        "raw_audit_artifact_sha256": sha256_file(audit_path),
        "merged_manifest_sha256": merged["manifest_sha256"],
        "merged_manifest_artifact_sha256": sha256_file(merge_path),
        "analysis_sha256": analysis["analysis_sha256"],
        "analysis_artifact_sha256": sha256_file(analysis_path),
        "confirmatory_status": analysis["gate"]["status"],
        "test_complete_seal_sha256": sha256_file(seal_path),
        "test_complete_sealed": True,
        "test_reopen_permitted": False,
        "frozen": True,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": payload["confirmatory_status"], "sealed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
