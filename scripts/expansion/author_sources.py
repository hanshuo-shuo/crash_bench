#!/usr/bin/env python3
"""Freeze source attempts before any simulator outcome or model score is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.source_registry import append_exposure_attempt


def _attempt_id(protocol_sha256: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(protocol_sha256.encode() + b"\0" + canonical).hexdigest()


def build_source_plan(config: dict[str, Any], *, mode: str, protocol_sha256: str) -> list[dict[str, Any]]:
    tasks = config.get("tasks", [])
    if not tasks:
        raise ValueError("source sampling config has no tasks")
    rows: list[dict[str, Any]] = []
    if mode == "screen":
        section = config["screen"]
        mechanisms = section["ordered_mechanisms"]
        per_cell = int(section["sources_per_mechanism_task"])
        base_seed = int(section["base_reset_seed"])
        ordinal = 0
        for mechanism in mechanisms:
            for task in tasks:
                for candidate_index in range(per_cell):
                    core = {
                        "cohort": "engineering_screen",
                        "mechanism_id": mechanism,
                        "suite": task["suite"],
                        "task_id": int(task["task_id"]),
                        "candidate_index": candidate_index,
                        "reset_seed": base_seed + ordinal,
                    }
                    rows.append({**core, "attempt_id": _attempt_id(protocol_sha256, core)})
                    ordinal += 1
    elif mode == "formal_nominal":
        section = config["formal_nominal"]
        if section.get("option_outcomes_allowed_during_enumeration") is not False:
            raise ValueError("formal nominal authoring must forbid option outcomes")
        if section.get("router_scores_allowed") is not False:
            raise ValueError("formal nominal authoring must forbid router scores")
        per_task = int(section["candidates_per_task"])
        base_seed = int(section["base_reset_seed"])
        ordinal = 0
        for task in tasks:
            for candidate_index in range(per_task):
                core = {
                    "cohort": "formal_nominal_candidate",
                    "mechanism_id": None,
                    "suite": task["suite"],
                    "task_id": int(task["task_id"]),
                    "candidate_index": candidate_index,
                    "reset_seed": base_seed + ordinal,
                }
                rows.append({**core, "attempt_id": _attempt_id(protocol_sha256, core)})
                ordinal += 1
    else:
        raise ValueError(f"unsupported source authoring mode: {mode}")
    if len({row["attempt_id"] for row in rows}) != len(rows):
        raise RuntimeError("source authoring produced duplicate attempt IDs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("screen", "formal_nominal"), required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--exposure-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.protocol_sha256) != 64:
        raise ValueError("protocol SHA-256 must contain 64 hex characters")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite source plan: {args.output}")
    config = json.loads(args.config.read_text())
    rows = build_source_plan(config, mode=args.mode, protocol_sha256=args.protocol_sha256)
    artifact_role = "ENGINEERING_SCREEN" if args.mode == "screen" else "EXPOSED_NOMINAL_AUTHORING"
    appended = 0
    for row in rows:
        appended += int(
            append_exposure_attempt(
                args.exposure_ledger,
                {
                    **row,
                    "artifact_role": artifact_role,
                    "protocol_sha256": args.protocol_sha256,
                },
            )
        )
    payload = {
        "schema_version": 1,
        "kind": "crashbench_expansion_source_plan",
        "mode": args.mode,
        "protocol_sha256": args.protocol_sha256,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_attempt_count": len(rows),
        "newly_appended_exposure_count": appended,
        "outcomes_opened": 0,
        "router_scores_read": 0,
        "attempts": rows,
    }
    payload["plan_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "attempts": len(rows), "appended": appended}, sort_keys=True))


if __name__ == "__main__":
    main()
