#!/usr/bin/env python3
"""Freeze deterministic, condition-stratified source splits for glass D0.

The assignment uses only source-state hashes and designed capture conditions.
It deliberately does not inspect crash outcomes, hidden states, or detector
scores, so the manifest can be committed before fitting or calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SPLITS = ("train", "calibration", "development")


def _condition(row: dict) -> str:
    value = str(row.get("condition", row.get("cond", ""))).lower()
    return {
        "on_path": "glass",
        "onpath": "glass",
        "off_path": "offpath",
        "no_glass": "noglass",
    }.get(value, value)


def _allocation(n_sources: int, fractions: tuple[float, float, float]) -> list[int]:
    if n_sources < len(SPLITS):
        raise ValueError(
            f"each condition stratum needs at least {len(SPLITS)} source states; "
            f"found {n_sources}"
        )
    raw = [n_sources * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    for index in sorted(
        range(len(SPLITS)), key=lambda item: raw[item] - counts[item], reverse=True
    )[:n_sources - sum(counts)]:
        counts[index] += 1
    for empty in [index for index, count in enumerate(counts) if count == 0]:
        donor = max(range(len(counts)), key=counts.__getitem__)
        if counts[donor] <= 1:
            raise ValueError("cannot make every split non-empty")
        counts[donor] -= 1
        counts[empty] += 1
    assert sum(counts) == n_sources and all(counts)
    return counts


def prepare(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {output}; pass --overwrite")
    rows = json.loads(Path(args.metadata).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("capture metadata must be a non-empty JSON row list")
    conditions_by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source = row.get("source_state_sha256")
        if not source:
            raise ValueError("capture metadata predates source_state_sha256; recapture D0")
        condition = _condition(row)
        if condition not in ("glass", "offpath", "noglass"):
            raise ValueError(f"unsupported capture condition {condition!r}")
        conditions_by_source[str(source)].add(condition)

    fractions = (
        float(args.train_fraction),
        float(args.calibration_fraction),
        1.0 - float(args.train_fraction) - float(args.calibration_fraction),
    )
    if any(fraction <= 0.0 for fraction in fractions):
        raise ValueError("train, calibration, and development fractions must be positive")
    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for source, conditions in conditions_by_source.items():
        strata[tuple(sorted(conditions))].append(source)

    source_splits = {}
    stratum_summary = []
    for signature, sources in sorted(strata.items()):
        ordered = sorted(
            sources,
            key=lambda source: hashlib.sha256(
                f"{args.seed}:{source}".encode("utf-8")
            ).hexdigest(),
        )
        counts = _allocation(len(ordered), fractions)
        cursor = 0
        assigned_counts = {}
        for split, count in zip(SPLITS, counts):
            assigned = ordered[cursor:cursor + count]
            source_splits.update({source: split for source in assigned})
            assigned_counts[split] = len(assigned)
            cursor += count
        stratum_summary.append({
            "conditions": list(signature),
            "source_states": len(ordered),
            "assigned": assigned_counts,
        })

    payload = {
        "schema_version": 1,
        "kind": "glass_detector_source_split",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_actions": int(args.horizon_actions),
        "assignment": {
            "method": "condition_stratified_hash_order",
            "uses_outcomes_or_model_scores": False,
            "seed": int(args.seed),
            "fractions": dict(zip(SPLITS, fractions)),
            "strata": stratum_summary,
        },
        "source_splits": dict(sorted(source_splits.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "sources": len(source_splits),
        "counts": {
            split: sum(value == split for value in source_splits.values())
            for split in SPLITS
        },
    }, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon-actions", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--overwrite", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
