#!/usr/bin/env python3
"""Create source-macro D5 coverage and realized-outcome machine tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


GROUP_FIELDS = ("split_role", "task_id", "condition", "severity_id", "option_id")
METRICS = (
    "u0", "task_success", "catastrophe", "safe_noncompletion", "intervention_invoked",
    "option_duration_steps", "path_length_m", "max_force_n", "force_exposure_ns", "latency_ms",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metric(row: dict[str, Any], key: str) -> float:
    return float(row["u0"] if key == "u0" else row["outcome"][key])


def source_macro_cells(branches: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in branches:
        grouped[tuple(str(row[field]) for field in GROUP_FIELDS)].append(row)
    cells = []
    for key, rows in sorted(grouped.items()):
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_source[str(row["physical_source_id"])].append(row)
        result = {field: value for field, value in zip(GROUP_FIELDS, key)}
        result.update(
            {
                "physical_source_count": len(by_source),
                "branch_count": len(rows),
                **{
                    f"source_macro_{name}": float(
                        np.mean([
                            np.mean([metric(row, name) for row in source_rows])
                            for source_rows in by_source.values()
                        ])
                    )
                    for name in METRICS
                },
            }
        )
        cells.append(result)
    return cells


def summarize(
    anchors: list[dict[str, Any]], branches: list[dict[str, Any]], invalid: list[dict[str, Any]]
) -> dict[str, Any]:
    roles = sorted(set(str(row["split_role"]) for row in anchors))
    if set(roles) - {"train", "calibration", "development"}:
        raise ValueError("D5 summary received a forbidden split role")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d5_statewise_machine_summary",
        "scope": "single_primary_policy_single_staleness_mechanism_pilot",
        "physical_source_count": len({row["physical_source_id"] for row in anchors}),
        "anchor_count": len(anchors),
        "branch_count": len(branches),
        "invalid_block_count": len(invalid),
        "role_source_counts": {
            role: len({row["physical_source_id"] for row in anchors if row["split_role"] == role})
            for role in roles
        },
        "task_source_counts": {
            task: len({row["physical_source_id"] for row in anchors if row["task_id"] == task})
            for task in sorted(set(row["task_id"] for row in anchors))
        },
        "cells": source_macro_cells(branches),
        "test_rows_read": 0,
    }
    payload["summary_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D5 summary: {args.output}")
    payload = summarize(
        read_jsonl(args.merged_dir / "anchors.jsonl"),
        read_jsonl(args.merged_dir / "branches.jsonl"),
        read_jsonl(args.merged_dir / "invalid_blocks.jsonl"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "sources": payload["physical_source_count"], "cells": len(payload["cells"])}, sort_keys=True))


if __name__ == "__main__":
    main()
