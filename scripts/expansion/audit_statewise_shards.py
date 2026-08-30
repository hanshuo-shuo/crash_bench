#!/usr/bin/env python3
"""Audit D5 source shards before any merge or model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.branching.artifacts import BlobRef, ContentAddressedStore


ALLOWED_ROLES = ("train", "calibration", "development")
ROLE_ORDER = {role: index for index, role in enumerate(ALLOWED_ROLES)}
EXPECTED_OPTIONS = {"base_continue", "observation_refresh", "safe_stop"}
REQUIRED_OUTCOME_FIELDS = {
    "task_success", "catastrophe", "safe_noncompletion", "intervention_invoked",
    "human_help", "option_duration_steps", "path_length_m", "max_force_n",
    "force_exposure_ns", "inference_latency_ms", "actuation_latency_ms", "latency_ms",
}


def active_assignments(split: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in split["assignments"] if row["role"] in ALLOWED_ROLES]
    rows.sort(
        key=lambda row: (
            str(row["task_id"]), ROLE_ORDER[str(row["role"])], str(row["physical_source_id"])
        )
    )
    if len(rows) != 48:
        raise ValueError("D5 split does not contain exactly 48 active assignments")
    return rows


def self_hash(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("shard_sha256", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit(
    shards: Iterable[dict[str, Any]],
    *,
    split: dict[str, Any],
    store: ContentAddressedStore,
    expected_commit: str,
) -> dict[str, Any]:
    rows = list(shards)
    assignments = active_assignments(split)
    errors: list[str] = []
    indices = [int(row.get("assignment_index", -1)) for row in rows]
    if sorted(indices) != list(range(48)):
        errors.append("assignment_indices_are_not_exactly_0_through_47")
    audits = []
    for shard in sorted(rows, key=lambda row: int(row.get("assignment_index", -1))):
        index = int(shard.get("assignment_index", -1))
        shard_errors = []
        if not 0 <= index < 48:
            shard_errors.append("invalid_assignment_index")
            audits.append({"assignment_index": index, "errors": shard_errors})
            continue
        assignment = assignments[index]
        for key, expected_key in (
            ("physical_source_id", "physical_source_id"),
            ("mechanism_source_id", "mechanism_source_id"),
            ("policy_source_id", "policy_source_id"),
            ("task_id", "task_id"),
            ("split_role", "role"),
        ):
            if shard.get(key) != assignment.get(expected_key):
                shard_errors.append(f"split_identity_mismatch:{key}")
        if shard.get("protocol_sha256") != split.get("protocol_sha256"):
            shard_errors.append("protocol_sha256_mismatch")
        if shard.get("execution", {}).get("git_commit") != expected_commit:
            shard_errors.append("execution_commit_mismatch")
        if shard.get("test_rows_read") != 0:
            shard_errors.append("test_rows_read_nonzero")
        if shard.get("planned_blocks") != 27 or shard.get("complete_blocks") != 27:
            shard_errors.append("incomplete_block_count")
        if shard.get("all_blocks_accounted") is not True or len(shard.get("blocks", [])) != 27:
            shard_errors.append("block_accounting_mismatch")
        if shard.get("shard_sha256") != self_hash(shard):
            shard_errors.append("shard_self_hash_mismatch")
        block_ids = []
        for block in shard.get("blocks", []):
            block_ids.append(str(block.get("block_id")))
            if block.get("status") != "COMPLETE_REALIZED_OPTIONS":
                shard_errors.append(f"noncomplete_block:{block.get('block_id')}")
                continue
            try:
                store.validate(BlobRef(**block["anchor_feature_blob"]))
            except Exception as exc:
                shard_errors.append(f"feature_blob:{block.get('block_id')}:{type(exc).__name__}")
            outcomes = block.get("option_outcomes", [])
            if len(outcomes) != 3 or {row.get("option_id") for row in outcomes} != EXPECTED_OPTIONS:
                shard_errors.append(f"option_completeness:{block.get('block_id')}")
                continue
            if len({row.get("branch_start_bundle_id") for row in outcomes}) != 1:
                shard_errors.append(f"bundle_start_mismatch:{block.get('block_id')}")
            if len({json.dumps(row.get("branch_start_component_hashes"), sort_keys=True) for row in outcomes}) != 1:
                shard_errors.append(f"component_start_mismatch:{block.get('block_id')}")
            if len({row.get("mechanism_queue_sha256") for row in outcomes}) != 1:
                shard_errors.append(f"queue_start_mismatch:{block.get('block_id')}")
            for outcome in outcomes:
                missing = REQUIRED_OUTCOME_FIELDS - set(outcome.get("outcome", {}))
                if missing:
                    shard_errors.append(
                        f"outcome_metric_missing:{block.get('block_id')}:{outcome.get('option_id')}:{','.join(sorted(missing))}"
                    )
        if len(block_ids) != len(set(block_ids)):
            shard_errors.append("duplicate_block_id")
        errors.extend(f"source_{index}:{error}" for error in shard_errors)
        audits.append(
            {
                "assignment_index": index,
                "physical_source_id": shard.get("physical_source_id"),
                "split_role": shard.get("split_role"),
                "task_id": shard.get("task_id"),
                "complete_blocks": shard.get("complete_blocks"),
                "errors": shard_errors,
            }
        )
    role_counts = Counter(str(row.get("split_role")) for row in rows)
    task_counts = Counter(str(row.get("task_id")) for row in rows)
    if role_counts != Counter({"train": 24, "calibration": 12, "development": 12}):
        errors.append("split_role_counts_mismatch")
    if task_counts != Counter({"libero_spatial:0": 24, "libero_spatial:2": 24}):
        errors.append("task_counts_mismatch")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d5_source_shard_audit",
        "expected_execution_commit": expected_commit,
        "protocol_sha256": split["protocol_sha256"],
        "observed_shards": len(rows),
        "role_counts": dict(sorted(role_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "complete_blocks": sum(int(row.get("complete_blocks", 0)) for row in rows),
        "errors": errors,
        "sources": audits,
        "test_rows_read": sum(int(row.get("test_rows_read", 0)) for row in rows),
        "status": "GO" if not errors else "NO_GO",
        "next_action": "MERGE_D5_DATASET" if not errors else "STOP_BEFORE_MERGE",
    }
    payload["audit_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D5 shard audit: {args.output}")
    payload = audit(
        (json.loads(path.read_text()) for path in args.shard),
        split=json.loads(args.split_manifest.read_text()),
        store=ContentAddressedStore(args.artifact_store),
        expected_commit=args.expected_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["status"], "shards": payload["observed_shards"]}, sort_keys=True))
    if payload["status"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
