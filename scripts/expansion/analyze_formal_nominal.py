#!/usr/bin/env python3
"""Audit 126 fresh candidates and select exactly 100 scene-unique formal sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def analyze_formal_shards(shards: Iterable[dict[str, Any]]) -> dict[str, Any]:
    shards = list(shards)
    indices = [int(row.get("shard_index", -1)) for row in shards]
    duplicate_shards = sorted({index for index in indices if indices.count(index) > 1})
    missing_shards = sorted(set(range(6)) - set(indices))
    rows = [row for shard in shards for row in shard.get("rows", [])]
    attempt_ids = [str(row.get("attempt_id")) for row in rows]
    source_hashes = [str(row.get("source_state_sha256")) for row in rows if row.get("source_state_sha256")]
    complete = [row for row in rows if row.get("status") == "NOMINAL_COMPLETE"]
    selected = []
    task_summary = {}
    for task_id in (0, 2):
        task_rows = sorted(
            [row for row in complete if int(row["task_id"]) == task_id],
            key=lambda row: int(row["candidate_index"]),
        )
        successful = [row for row in task_rows if row.get("task_success") is True]
        seen_fingerprints = set()
        unique_successful = []
        for row in successful:
            fingerprint = str(row["scene_fingerprint"])
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            unique_successful.append(row)
        chosen = unique_successful[:50]
        for row in chosen:
            identity = {
                "task_id": task_id,
                "source_state_sha256": row["source_state_sha256"],
                "scene_fingerprint": row["scene_fingerprint"],
                "reset_seed": row["reset_seed"],
            }
            physical_id = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            selected.append(
                {
                    **identity,
                    "physical_source_id": physical_id,
                    "candidate_index": row["candidate_index"],
                    "attempt_id": row["attempt_id"],
                    "source_state_dtype": row["source_state_dtype"],
                    "source_state": row["source_state"],
                    "role": "FORMAL_SOURCE_PRE_SPLIT",
                }
            )
        task_summary[str(task_id)] = {
            "attempted": sum(int(row["task_id"]) == task_id for row in rows),
            "complete": len(task_rows),
            "successful": len(successful),
            "unique_successful": len(unique_successful),
            "selected": len(chosen),
            "nominal_success_rate": len(successful) / len(task_rows) if task_rows else 0,
        }
    hard_failures = []
    if duplicate_shards or missing_shards:
        hard_failures.append("shard_identity")
    if len(rows) != 126 or len(complete) != 126:
        hard_failures.append("candidate_completeness")
    if len(set(attempt_ids)) != len(attempt_ids):
        hard_failures.append("duplicate_attempt_id")
    if len(set(source_hashes)) != len(source_hashes):
        hard_failures.append("duplicate_exact_source_state")
    if any(row.get("option_outcomes_opened") != 0 for row in rows):
        hard_failures.append("option_outcome_leakage")
    for task_id in (0, 2):
        summary = task_summary[str(task_id)]
        if summary["nominal_success_rate"] < 0.70:
            hard_failures.append(f"task{task_id}_nominal_rate")
        if summary["unique_successful"] < 50:
            hard_failures.append(f"task{task_id}_unique_successful_support")
    if len(selected) != 100 or len({row["physical_source_id"] for row in selected}) != 100:
        hard_failures.append("selected_physical_source_count")
    status = "GO" if not hard_failures else "NO_GO"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_formal_nominal_analysis",
        "observed_shards": len(shards),
        "duplicate_shards": duplicate_shards,
        "missing_shards": missing_shards,
        "attempted_candidates": len(rows),
        "complete_candidates": len(complete),
        "task_summary": task_summary,
        "selected_sources": selected,
        "selected_physical_sources": len(selected),
        "hard_failures": hard_failures,
        "status": status,
        "next_action": "FREEZE_D4_SPLITS" if status == "GO" else "STOP_FORMAL_SOURCE_FREEZE",
        "option_outcomes_opened": 0,
    }
    payload["analysis_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite formal nominal analysis: {args.output}")
    payload = analyze_formal_shards(json.loads(path.read_text()) for path in args.shard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["status"], "selected": payload["selected_physical_sources"]}, sort_keys=True))


if __name__ == "__main__":
    main()
