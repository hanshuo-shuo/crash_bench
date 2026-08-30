#!/usr/bin/env python3
"""Audit all once-authorized D8-B shards before confirmatory analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.branching.artifacts import ContentAddressedStore
from scripts.expansion.audit_statewise_shards import audit
from scripts.expansion.collect_staleness_statewise import test_assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--authorization-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D8-B shard audit: {args.output}")
    authorization = json.loads((args.authorization_root / "test_authorization.json").read_text())
    shards = [json.loads(path.read_text()) for path in args.shard]
    if any(row.get("authorization_token") != authorization["authorization_token"] for row in shards):
        raise ValueError("D8-B shard authorization token drift")
    split = json.loads(args.split_manifest.read_text())
    payload = audit(
        shards,
        split=split,
        store=ContentAddressedStore(args.artifact_store),
        expected_commit=args.expected_commit,
        assignments_override=test_assignments(split),
        expected_role_counts=Counter({"confirmatory_id_test": 32}),
        expected_task_counts=Counter({"libero_spatial:0": 16, "libero_spatial:2": 16}),
        expected_test_rows_read_per_shard=1,
        kind="crashbench_expansion_d8_benchmark_test_shard_audit",
    )
    payload["authorization_token"] = authorization["authorization_token"]
    payload.pop("audit_sha256", None)
    payload["audit_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["status"], "shards": payload["observed_shards"]}, sort_keys=True))
    if payload["status"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
