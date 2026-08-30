#!/usr/bin/env python3
"""Audit the eight nominal-only D2/D3 preflight cells and issue Gate A0 readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from crashbench.governance.gates import Criterion, CriterionClass, GatePolicy, evaluate_gate
from scripts.expansion.run_nominal_preflight import CELL_ORDER


def analyze_cells(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates = []
    for payload in payloads:
        cell = (str(payload.get("mechanism_id")), int(payload.get("task_id", -1)))
        if cell in by_cell:
            duplicates.append(cell)
        by_cell[cell] = payload
    cell_rows = []
    for mechanism, task_id in CELL_ORDER:
        payload = by_cell.get((mechanism, task_id))
        if payload is None:
            cell_rows.append(
                {
                    "mechanism_id": mechanism,
                    "task_id": task_id,
                    "observed": False,
                    "attempted_sources": 0,
                    "complete_sources": 0,
                    "nominal_successes": 0,
                    "nominal_success_rate": None,
                    "path_ready_sources": 0,
                }
            )
            continue
        rows = payload.get("rows", [])
        path_ready = sum(
            row.get("status") == "NOMINAL_COMPLETE"
            and float(row.get("eef_path_length_m", 0)) >= 0.10
            and len(row.get("eef_path_xyz", [])) >= 2
            for row in rows
        )
        cell_rows.append(
            {
                "mechanism_id": mechanism,
                "task_id": task_id,
                "observed": True,
                "attempted_sources": int(payload.get("attempted_sources", -1)),
                "complete_sources": int(payload.get("complete_sources", -1)),
                "nominal_successes": int(payload.get("nominal_successes", -1)),
                "nominal_success_rate": float(payload.get("nominal_success_rate", -1)),
                "path_ready_sources": int(path_ready),
                "all_attempts_accounted": payload.get("all_attempts_accounted"),
                "option_outcomes_opened": payload.get("option_outcomes_opened"),
                "cell_sha256": payload.get("cell_sha256"),
                "slurm_job_id": payload.get("execution", {}).get("slurm_job_id"),
                "slurm_array_task_id": payload.get("execution", {}).get("slurm_array_task_id"),
            }
        )
    criteria = [
        Criterion(
            "duplicate_cell_count", CriterionClass.HARD_VALIDITY,
            len(duplicates), "==", 0
        )
    ]
    for row in cell_rows:
        prefix = f"{row['mechanism_id']}:task{row['task_id']}"
        criteria.extend(
            [
                Criterion(
                    f"{prefix}:observed", CriterionClass.HARD_VALIDITY,
                    row["observed"], "==", True
                ),
                Criterion(
                    f"{prefix}:attempted", CriterionClass.HARD_VALIDITY,
                    row["attempted_sources"], "==", 8
                ),
                Criterion(
                    f"{prefix}:complete", CriterionClass.HARD_VALIDITY,
                    row["complete_sources"], "==", 8
                ),
                Criterion(
                    f"{prefix}:accounted", CriterionClass.HARD_VALIDITY,
                    row.get("all_attempts_accounted"), "==", True
                ),
                Criterion(
                    f"{prefix}:no_option_outcomes", CriterionClass.HARD_VALIDITY,
                    row.get("option_outcomes_opened"), "==", 0
                ),
                Criterion(
                    f"{prefix}:nominal_success", CriterionClass.CLAIM_SCOPE,
                    row["nominal_success_rate"], ">=", 0.70
                ),
                Criterion(
                    f"{prefix}:path_ready", CriterionClass.CLAIM_SCOPE,
                    row["path_ready_sources"], ">=", 6
                ),
            ]
        )
    gate = evaluate_gate(
        criteria,
        GatePolicy(
            stage_id="D2_D3_NOMINAL_PREFLIGHT_A0",
            confirmatory=False,
            test_outcomes_opened=False,
            allow_scoped_continuation=True,
            go_next_action="DERIVE_MECHANICAL_INJECTIONS_AND_OPEN_NON_UNSTABLE_D2_SCREENS",
            scoped_next_action="CONTINUE_ONLY_PASSING_CELLS_OR_REPLACE_TASK_BEFORE_MECHANISM_OUTCOMES",
            fail_next_action="STOP_PREFLIGHT_AND_RESOLVE_ACCOUNTING_OR_OPERATIONAL_FAILURES",
        ),
    )
    task_summary = {}
    for task_id in (0, 2):
        rows = [row for row in cell_rows if row["task_id"] == task_id]
        successes = sum(row["nominal_successes"] for row in rows)
        attempts = sum(row["attempted_sources"] for row in rows)
        task_summary[str(task_id)] = {
            "attempted_sources": attempts,
            "nominal_successes": successes,
            "nominal_success_rate": successes / attempts if attempts else None,
        }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_nominal_preflight_analysis",
        "expected_cells": len(CELL_ORDER),
        "observed_cells": sum(row["observed"] for row in cell_rows),
        "duplicate_cells": [list(cell) for cell in duplicates],
        "cells": cell_rows,
        "task_summary": task_summary,
        "gate": gate,
        "option_outcomes_opened": 0,
    }
    payload["analysis_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite nominal analysis: {args.output}")
    payload = analyze_cells(json.loads(path.read_text()) for path in args.cell)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["gate"]["status"], "next_action": payload["gate"]["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
