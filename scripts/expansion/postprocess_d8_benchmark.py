#!/usr/bin/env python3
"""Audit, merge, analyze, and seal the single authorized D8-B test run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.splits import seal_test_complete
from scripts.expansion.hash_tree_manifest import resolve_git_head


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--benchmark-freeze", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    authorization_root = args.run_root / "authorization"
    if (authorization_root / "test_complete.seal").exists():
        raise FileExistsError("D8-B test is already sealed")
    post = args.run_root / "postprocess"
    if post.exists():
        raise FileExistsError("refusing to overwrite D8-B postprocess root")
    shards = sorted((args.run_root / "shards").glob("source_*.json"))
    if len(shards) != 32:
        raise ValueError(f"D8-B postprocess requires 32 shards, found {len(shards)}")
    post.mkdir(parents=True)
    shard_args = [item for path in shards for item in ("--shard", str(path))]
    audit_path = post / "raw_shard_audit.json"
    run([
        sys.executable, "scripts/expansion/audit_benchmark_test_shards.py", *shard_args,
        "--split-manifest", str(args.split_manifest),
        "--artifact-store", str(args.run_root / "artifact_store"),
        "--authorization-root", str(authorization_root),
        "--expected-commit", args.expected_commit,
        "--output", str(audit_path),
    ])
    merged = post / "merged"
    run([
        sys.executable, "scripts/expansion/merge_benchmark_test_dataset.py", *shard_args,
        "--artifact-store", str(args.run_root / "artifact_store"),
        "--utility-config", str(args.utility_config),
        "--authorization-root", str(authorization_root),
        "--output-dir", str(merged),
    ])
    analysis_path = post / "benchmark_test_analysis.json"
    run([
        sys.executable, "scripts/expansion/analyze_benchmark_test.py",
        "--merged-dir", str(merged),
        "--utility-config", str(args.utility_config),
        "--benchmark-freeze", str(args.benchmark_freeze),
        "--output", str(analysis_path),
    ])
    audit = json.loads(audit_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    seal_test_complete(
        authorization_root, completeness_audit_sha256=audit["audit_sha256"]
    )
    manifest = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d8_benchmark_postprocess_manifest",
        "execution_git_commit": args.expected_commit,
        "postprocess_git_commit": resolve_git_head(ROOT),
        "physical_source_shards": 32,
        "test_sources_read": 32,
        "method_superiority_evaluated": False,
        "raw_audit_sha256": audit["audit_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "confirmatory_status": analysis["gate"]["status"],
        "test_complete_sealed": True,
    }
    (post / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(post), "status": manifest["confirmatory_status"], "sealed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
