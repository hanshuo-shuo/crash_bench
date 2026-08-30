#!/usr/bin/env python3
"""Audit pre-outcome mechanical derivation across all eight authorized cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.governance.gates import Criterion, CriterionClass, GatePolicy, evaluate_gate
from scripts.expansion.run_nominal_preflight import CELL_ORDER


def analyze_mechanical_cells(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_cell = {}
    duplicates = []
    for payload in payloads:
        cell = (str(payload.get("mechanism_id")), int(payload.get("task_id", -1)))
        if cell in by_cell:
            duplicates.append(cell)
        by_cell[cell] = payload
    criteria = [
        Criterion("duplicate_cell_count", CriterionClass.HARD_VALIDITY, len(duplicates), "==", 0)
    ]
    rows = []
    for mechanism, task_id in CELL_ORDER:
        payload = by_cell.get((mechanism, task_id))
        observed = payload is not None
        attempted = int(payload.get("attempted_sources", 0)) if observed else 0
        valid = int(payload.get("mechanically_valid_sources", 0)) if observed else 0
        accounted = payload.get("all_attempts_accounted") if observed else None
        outcomes = payload.get("option_outcomes_opened") if observed else None
        severity_invalid = 0
        if observed:
            for source in payload.get("rows", []):
                for severity in source.get("severities", {}).values():
                    if severity.get("validity", {}).get("valid") is not True:
                        severity_invalid += 1
        prefix = f"{mechanism}:task{task_id}"
        criteria.extend(
            [
                Criterion(f"{prefix}:observed", CriterionClass.HARD_VALIDITY, observed, "==", True),
                Criterion(f"{prefix}:attempted", CriterionClass.HARD_VALIDITY, attempted, "==", 8),
                Criterion(f"{prefix}:accounted", CriterionClass.HARD_VALIDITY, accounted, "==", True),
                Criterion(f"{prefix}:no_option_outcomes", CriterionClass.HARD_VALIDITY, outcomes, "==", 0),
                Criterion(f"{prefix}:severity_invalid", CriterionClass.HARD_VALIDITY, severity_invalid, "==", 0),
                Criterion(f"{prefix}:valid_sources", CriterionClass.CLAIM_SCOPE, valid, ">=", 6),
            ]
        )
        rows.append(
            {
                "mechanism_id": mechanism,
                "task_id": task_id,
                "observed": observed,
                "attempted_sources": attempted,
                "mechanically_valid_sources": valid,
                "severity_invalid_count": severity_invalid,
                "cell_sha256": payload.get("cell_sha256") if observed else None,
                "slurm_job_id": payload.get("execution", {}).get("slurm_job_id") if observed else None,
                "slurm_array_task_id": payload.get("execution", {}).get("slurm_array_task_id") if observed else None,
            }
        )
    gate = evaluate_gate(
        criteria,
        GatePolicy(
            stage_id="D2_MECHANICAL_PREFLIGHT",
            confirmatory=False,
            test_outcomes_opened=False,
            allow_scoped_continuation=True,
            go_next_action="OPEN_NON_UNSTABLE_D2_OUTCOME_SCREENS",
            scoped_next_action="OPEN_ONLY_MECHANICALLY_VALID_CELLS_WITH_SCOPED_CLAIMS",
            fail_next_action="STOP_D2_OUTCOMES_AND_RESOLVE_MECHANICAL_ACCOUNTING",
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_mechanical_preflight_analysis",
        "expected_cells": 8,
        "observed_cells": sum(row["observed"] for row in rows),
        "duplicate_cells": [list(cell) for cell in duplicates],
        "cells": rows,
        "gate": gate,
        "option_outcomes_opened": 0,
        "unstable_v2_authorized": False,
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
        raise FileExistsError(f"refusing to overwrite mechanical analysis: {args.output}")
    payload = analyze_mechanical_cells(json.loads(path.read_text()) for path in args.cell)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["gate"]["status"], "next_action": payload["gate"]["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
