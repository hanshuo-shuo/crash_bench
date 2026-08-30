#!/usr/bin/env python3
"""Audit all D1 cells and decide primary versus cross-policy capability scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


EXPECTED_CELLS = (
    ("openvla", 0),
    ("openvla", 2),
    ("pi0", 0),
    ("pi0", 2),
)


def audit_cells(
    payloads: Iterable[dict[str, Any]], *, primary_backend: str = "openvla"
) -> dict[str, Any]:
    if primary_backend not in {"openvla", "pi0"}:
        raise ValueError(f"unsupported primary backend: {primary_backend}")
    by_cell: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates = []
    for payload in payloads:
        cell = (str(payload.get("backend")), int(payload.get("source", {}).get("task_id", -1)))
        if cell in by_cell:
            duplicates.append(cell)
        by_cell[cell] = payload
    cell_rows = []
    for backend, task_id in EXPECTED_CELLS:
        payload = by_cell.get((backend, task_id))
        if payload is None:
            cell_rows.append(
                {
                    "backend": backend,
                    "task_id": task_id,
                    "observed": False,
                    "pass": None,
                    "classification": None,
                    "hard_failures": [],
                }
            )
            continue
        gate = payload.get("gate", {})
        criteria = gate.get("criteria", {})
        failures = []
        if gate.get("status") != "PASS":
            failures.append("gate_status")
        if gate.get("classification") != "DETERMINISTIC_EXACT":
            failures.append("classification")
        if not criteria or not all(value is True for value in criteria.values()):
            failures.append("repeat_criteria")
        if payload.get("source", {}).get("role") != "EXPOSED_ENGINEERING_ONLY":
            failures.append("source_role")
        if payload.get("claim_boundary") != (
            "Engineering conformance only; no scientific option outcome was opened."
        ):
            failures.append("claim_boundary")
        repeats = payload.get("repeats", [])
        if len(repeats) != 3:
            failures.append("repeat_count")
        cell_rows.append(
            {
                "backend": backend,
                "task_id": task_id,
                "observed": True,
                "pass": not failures,
                "classification": gate.get("classification"),
                "hard_failures": failures,
                "result_sha256": hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "slurm_job_id": payload.get("execution", {}).get("slurm_job_id"),
                "execution_commit": payload.get("execution", {}).get("git_commit"),
                "capture_seconds": payload.get("bundle", {}).get("capture_seconds"),
                "restore_trace_seconds": [
                    row.get("restore_and_trace_seconds") for row in repeats
                ],
            }
        )
    if duplicates:
        status = "INVALID_DUPLICATE_CELL"
        next_action = "RESOLVE_DUPLICATE_D1_ARTIFACTS"
    else:
        transfer_backend = "pi0" if primary_backend == "openvla" else "openvla"
        primary = [row for row in cell_rows if row["backend"] == primary_backend]
        transfer = [row for row in cell_rows if row["backend"] == transfer_backend]
        primary_missing = any(row["pass"] is None for row in primary)
        transfer_missing = any(row["pass"] is None for row in transfer)
        primary_failed = any(row["pass"] is False for row in primary)
        transfer_failed = any(row["pass"] is False for row in transfer)
        if primary_missing:
            status = "INCOMPLETE_PRIMARY_FAIL_CLOSED"
            next_action = "COMPLETE_OPENVLA_D1_CELLS"
        elif primary_failed:
            status = "PRIMARY_BACKEND_NO_GO"
            next_action = "STOP_MCV_METHOD_AND_AUDIT_EXACTNESS_FAILURE"
        elif transfer_missing:
            status = "PRIMARY_GO_TRANSFER_PENDING"
            next_action = "CONTINUE_D2_D3_PRIMARY_ONLY_WHILE_COMPLETING_PI0_D1"
        elif transfer_failed:
            status = "PRIMARY_GO_SINGLE_POLICY_SCOPE"
            next_action = "CONTINUE_PRIMARY_D2_D3__DROP_MULTIPOLICY_CLAIM"
        else:
            status = "PRIMARY_AND_TRANSFER_GO"
            next_action = "CONTINUE_D2_D3_WITH_PI0_TRANSFER_TRACK"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d1_conformance_decision",
        "cells": cell_rows,
        "primary_backend": primary_backend,
        "duplicate_cells": [list(cell) for cell in duplicates],
        "status": status,
        "next_action": next_action,
        "primary_policy_gate_is_hard": True,
        "transfer_failure_changes_scope_not_primary_validity": True,
        "scientific_outcomes_opened": 0,
    }
    payload["decision_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-backend", choices=("openvla", "pi0"), default="openvla")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D1 audit: {args.output}")
    payloads = [json.loads(path.read_text()) for path in args.result]
    decision = audit_cells(payloads, primary_backend=args.primary_backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": decision["status"], "next_action": decision["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
