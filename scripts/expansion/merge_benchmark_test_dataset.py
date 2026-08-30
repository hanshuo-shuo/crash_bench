#!/usr/bin/env python3
"""Merge the once-authorized 32-source D8-B benchmark outcome dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.branching.artifacts import ContentAddressedStore
from scripts.expansion.collect_staleness_statewise import TEST_ROLE
from scripts.expansion.merge_statewise_dataset import (
    _write_jsonl,
    budgets_from_config,
    merge_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--authorization-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite D8-B merge: {args.output_dir}")
    authorization = json.loads((args.authorization_root / "test_authorization.json").read_text())
    if not (args.authorization_root / "test_open.lock").is_file():
        raise RuntimeError("D8-B merge requires the still-open one-time test lock")
    shards = [json.loads(path.read_text()) for path in args.shard]
    if any(row.get("authorization_token") != authorization["authorization_token"] for row in shards):
        raise ValueError("D8-B shard authorization token drift")
    manifest, anchors, branches, invalid = merge_shards(
        shards,
        store=ContentAddressedStore(args.artifact_store),
        budgets=budgets_from_config(json.loads(args.utility_config.read_text())),
        expected_indices=tuple(range(32)),
        allowed_roles={TEST_ROLE},
        expected_test_rows_read_per_shard=1,
        kind="crashbench_expansion_d8_benchmark_test_merged_manifest",
    )
    if manifest["physical_source_count"] != 32:
        manifest["errors"].append("physical_source_count_is_not_32")
        manifest["status"] = "NO_GO"
    manifest["authorization_token"] = authorization["authorization_token"]
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "anchors.jsonl", anchors)
    _write_jsonl(args.output_dir / "branches.jsonl", branches)
    _write_jsonl(args.output_dir / "invalid_blocks.jsonl", invalid)
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output_dir), "status": manifest["status"], "sources": manifest["physical_source_count"]}, sort_keys=True))
    if manifest["status"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
