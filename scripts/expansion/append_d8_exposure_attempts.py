#!/usr/bin/env python3
"""Append the once-opened D8-B source outcomes to the immutable exposure ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def canonical_sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--test-complete-seal", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    shards = [json.loads(path.read_text()) for path in args.shard]
    if len(shards) != 32 or len({row["physical_source_id"] for row in shards}) != 32:
        raise ValueError("D8 exposure append requires exactly 32 physical-source shards")
    authorization = json.loads(args.authorization.read_text())
    seal = json.loads(args.test_complete_seal.read_text())
    if seal.get("kind") != "crashbench_test_complete_seal":
        raise ValueError("D8 exposure append requires the test completion seal")
    if any(row.get("authorization_token") != authorization["authorization_token"] for row in shards):
        raise ValueError("D8 exposure shard authorization drift")
    existing_rows = [
        json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()
    ]
    existing_ids = {row["attempt_id"] for row in existing_rows}
    rows = []
    for shard in sorted(shards, key=lambda row: row["physical_source_id"]):
        identity = {
            "run_id": authorization["run_id"],
            "authorization_token": authorization["authorization_token"],
            "physical_source_id": shard["physical_source_id"],
            "policy_source_id": shard["policy_source_id"],
            "event": "D8_B_CONFIRMATORY_OPTION_OUTCOMES_OPENED",
        }
        attempt_id = canonical_sha(identity)
        if attempt_id in existing_ids:
            raise ValueError(f"D8 exposure attempt already exists: {attempt_id}")
        rows.append(
            {
                "schema_version": 1,
                "kind": "crashbench_expansion_exposure_attempt",
                "attempt_id": attempt_id,
                "artifact_role": "EXPOSED_CONFIRMATORY_TEST_SCOPE_FAILURE",
                "cohort": "d8_benchmark_confirmatory_test",
                "protocol_sha256": shard["protocol_sha256"],
                "mechanism_id": "observation_staleness_v1",
                "suite": "libero_spatial",
                "task_id": int(str(shard["task_id"]).rsplit(":", 1)[-1]),
                "physical_source_id": shard["physical_source_id"],
                "mechanism_source_id": shard["mechanism_source_id"],
                "policy_source_id": shard["policy_source_id"],
                "authorization_token": authorization["authorization_token"],
                "test_complete_audit_sha256": seal["completeness_audit_sha256"],
                "option_outcomes_opened": True,
                "test_eligible": False,
            }
        )
    descriptor = os.open(args.ledger, os.O_WRONLY | os.O_APPEND)
    try:
        payload = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ).encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({"ledger": str(args.ledger), "appended": len(rows), "test_eligible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
