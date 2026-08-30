#!/usr/bin/env python3
"""Validate and merge D5 source shards into canonical anchor/branch tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.branching.artifacts import BlobRef, ContentAddressedStore
from crashbench.data.utility import (
    OutcomeVector,
    PhysicalBudgets,
    UtilityWeights,
    scalar_utility,
)


EXPECTED_OPTIONS = {"base_continue", "observation_refresh", "safe_stop"}
ALLOWED_ROLES = {"train", "calibration", "development"}


def budgets_from_config(config: dict[str, Any]) -> PhysicalBudgets:
    row = config["normalization"]
    return PhysicalBudgets(
        option_duration_steps=float(row["option_duration_steps"]),
        path_length_m=float(row["path_length_m"]),
        force_exposure_ns=float(row["force_exposure_ns"]),
        latency_ms=float(row["latency_ms"]),
        source=str(row["source"]),
    )


def outcome_vector(row: dict[str, Any]) -> OutcomeVector:
    return OutcomeVector(
        task_success=int(row["task_success"]),
        catastrophe=int(row["catastrophe"]),
        safe_noncompletion=int(row["safe_noncompletion"]),
        intervention_invoked=int(row["intervention_invoked"]),
        human_help=int(row["human_help"]),
        option_duration_steps=float(row["option_duration_steps"]),
        path_length_m=float(row["path_length_m"]),
        force_exposure_ns=float(row["force_exposure_ns"]),
        latency_ms=float(row["latency_ms"]),
    )


def merge_shards(
    shards: Iterable[dict[str, Any]],
    *,
    store: ContentAddressedStore,
    budgets: PhysicalBudgets,
    expected_indices: Sequence[int] = tuple(range(48)),
) -> tuple[dict[str, Any], list[dict], list[dict], list[dict]]:
    shards = list(shards)
    indices = [int(row.get("assignment_index", -1)) for row in shards]
    expected = set(map(int, expected_indices))
    duplicate = sorted({index for index in indices if indices.count(index) > 1})
    missing = sorted(expected - set(indices))
    unexpected = sorted(set(indices) - expected)
    anchors, branches, invalid = [], [], []
    errors = []
    physical_ids = set()
    for shard in sorted(shards, key=lambda row: int(row.get("assignment_index", -1))):
        role = str(shard.get("split_role"))
        if role not in ALLOWED_ROLES or "test" in role:
            errors.append(f"forbidden_split_role:{role}")
        if shard.get("test_rows_read") != 0:
            errors.append(f"test_rows_read:{shard.get('assignment_index')}")
        if shard.get("planned_blocks") != 27 or shard.get("all_blocks_accounted") is not True:
            errors.append(f"block_accounting:{shard.get('assignment_index')}")
        physical_id = str(shard.get("physical_source_id"))
        if physical_id in physical_ids:
            errors.append(f"duplicate_physical_source:{physical_id}")
        physical_ids.add(physical_id)
        for block in shard.get("blocks", []):
            base = {
                "block_id": block["block_id"],
                "physical_source_id": physical_id,
                "mechanism_source_id": shard["mechanism_source_id"],
                "policy_source_id": shard["policy_source_id"],
                "task_id": shard["task_id"],
                "split_role": role,
                "anchor_steps": block["anchor_steps"],
                "severity_id": block["severity_id"],
                "condition": block["condition"],
            }
            if block.get("status") == "PRE_ANCHOR_MECHANICAL_INVALID":
                invalid.append({**base, "status": block["status"], "reason": block["reason"]})
                continue
            if block.get("status") != "COMPLETE_REALIZED_OPTIONS":
                errors.append(f"nonterminal_block:{block['block_id']}:{block.get('status')}")
                invalid.append({**base, "status": block.get("status"), "reason": block.get("reason")})
                continue
            feature_ref = BlobRef(**block["anchor_feature_blob"])
            try:
                store.validate(feature_ref)
            except Exception as exc:
                errors.append(f"feature_blob:{block['block_id']}:{type(exc).__name__}")
            outcomes = block.get("option_outcomes", [])
            if {row.get("option_id") for row in outcomes} != EXPECTED_OPTIONS or len(outcomes) != 3:
                errors.append(f"option_completeness:{block['block_id']}")
                continue
            exact_start = (
                len({row["branch_start_bundle_id"] for row in outcomes}) == 1
                and len(
                    {
                        json.dumps(row["branch_start_component_hashes"], sort_keys=True)
                        for row in outcomes
                    }
                )
                == 1
                and len({row["mechanism_queue_sha256"] for row in outcomes}) == 1
            )
            if not exact_start:
                errors.append(f"branch_start_mismatch:{block['block_id']}")
            anchors.append(
                {
                    **base,
                    "bundle_id": block["bundle_id"],
                    "mechanism_queue_sha256": block["mechanism_queue_sha256"],
                    "anchor_feature_blob": block["anchor_feature_blob"],
                    "exact_branch_start": exact_start,
                }
            )
            for outcome in sorted(outcomes, key=lambda row: row["option_id"]):
                vector = outcome_vector(outcome["outcome"])
                branches.append(
                    {
                        **base,
                        "option_id": outcome["option_id"],
                        "terminal_signature": outcome["terminal_signature"],
                        "outcome": outcome["outcome"],
                        "u0": scalar_utility(vector, budgets),
                    }
                )
    if duplicate:
        errors.append("duplicate_assignment_indices")
    if missing:
        errors.append("missing_assignment_indices")
    if unexpected:
        errors.append("unexpected_assignment_indices")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d5_merged_dataset_manifest",
        "observed_source_shards": len(shards),
        "expected_source_shards": len(expected),
        "duplicate_assignment_indices": duplicate,
        "missing_assignment_indices": missing,
        "unexpected_assignment_indices": unexpected,
        "physical_source_count": len(physical_ids),
        "anchor_count": len(anchors),
        "branch_count": len(branches),
        "invalid_block_count": len(invalid),
        "errors": errors,
        "status": "GO" if not errors else "NO_GO",
        "test_rows_read": 0,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest, anchors, branches, invalid


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite merged dataset: {args.output_dir}")
    utility = json.loads(args.utility_config.read_text())
    manifest, anchors, branches, invalid = merge_shards(
        (json.loads(path.read_text()) for path in args.shard),
        store=ContentAddressedStore(args.artifact_store),
        budgets=budgets_from_config(utility),
    )
    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "anchors.jsonl", anchors)
    _write_jsonl(args.output_dir / "branches.jsonl", branches)
    _write_jsonl(args.output_dir / "invalid_blocks.jsonl", invalid)
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output_dir), "status": manifest["status"], "anchors": len(anchors), "branches": len(branches)}, sort_keys=True))
    if manifest["status"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
