#!/usr/bin/env python3
"""Resolve strict broad and scoped D2 mechanism selection without threshold rewriting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


ORDER = (
    "fragile_path_collision_v2",
    "unstable_final_placement_v2",
    "observation_staleness_v1",
    "action_drift_v1",
    "narrow_clearance_v1",
)


def resolve_selection(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_mechanism = {}
    duplicates = []
    for result in results:
        mechanism = str(result.get("mechanism_id"))
        if mechanism in by_mechanism:
            duplicates.append(mechanism)
        by_mechanism[mechanism] = result
    rows = []
    for mechanism in ORDER:
        if mechanism == "unstable_final_placement_v2":
            rows.append(
                {
                    "mechanism_id": mechanism,
                    "status": "NOT_RUN_REQUIRES_EXPLICIT_AUTHORIZATION",
                    "hard_validity_pass": None,
                    "formal_shortlist_eligible": False,
                }
            )
            continue
        result = by_mechanism.get(mechanism)
        if result is None:
            rows.append(
                {
                    "mechanism_id": mechanism,
                    "status": "MISSING",
                    "hard_validity_pass": None,
                    "formal_shortlist_eligible": False,
                }
            )
            continue
        gate = result.get("gate", {})
        status = gate.get("status")
        rows.append(
            {
                "mechanism_id": mechanism,
                "status": status,
                "hard_validity_pass": not gate.get("hard_failures")
                and not gate.get("missing_evidence"),
                "formal_shortlist_eligible": status == "GO",
                "claim_scope_failures": gate.get("claim_scope_failures", []),
                "analysis_sha256": result.get("analysis_sha256"),
            }
        )
    if duplicates:
        status = "INVALID_DUPLICATE_MECHANISM_RESULT"
        next_action = "RESOLVE_D2_RESULT_IDENTITY"
    elif any(row["status"] == "MISSING" for row in rows if "unstable" not in row["mechanism_id"]):
        status = "INCOMPLETE"
        next_action = "COMPLETE_AUTHORIZED_D2_SCREENS"
    else:
        go = [row for row in rows if row["status"] == "GO"]
        scoped = [row for row in rows if row["status"] == "SCOPED_CONTINUE"]
        hard_valid = [row for row in rows if row["hard_validity_pass"] is True]
        if len(go) >= 3:
            status = "BROAD_GATE_A0_GO"
            next_action = "FREEZE_THREE_MECHANISM_FORMAL_PROTOCOL"
        elif len(go) >= 2:
            status = "SCOPED_TWO_MECHANISM_CONTINUE"
            next_action = "FREEZE_TWO_MECHANISM_PROTOCOL__NO_BROAD_THREE_MECHANISM_CLAIM"
        elif len(go) >= 1 and len(hard_valid) >= 3:
            status = "SINGLE_FORMAL_PLUS_EXPLORATORY_METHOD_PILOT"
            next_action = "CONTINUE_LABELLED_METHOD_PILOT__NO_BROAD_BENCHMARK_CLAIM"
        else:
            status = "D2_NO_GO"
            next_action = "STOP_BROAD_METHOD_AND_RELEASE_MECHANISM_DIAGNOSTICS"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d2_selection_decision",
        "ordered_mechanisms": rows,
        "duplicate_mechanisms": sorted(set(duplicates)),
        "formal_go_count": sum(row["status"] == "GO" for row in rows),
        "scoped_continue_count": sum(row["status"] == "SCOPED_CONTINUE" for row in rows),
        "strict_broad_threshold": 3,
        "strict_broad_gate_pass": sum(row["status"] == "GO" for row in rows) >= 3,
        "status": status,
        "next_action": next_action,
        "unstable_v2_authorized": False,
        "claim_boundary": "Scoped continuation changes claim breadth, never a mechanism's frozen gate result.",
    }
    payload["decision_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D2 selection: {args.output}")
    payload = resolve_selection(json.loads(path.read_text()) for path in args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["status"], "next_action": payload["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
