#!/usr/bin/env python3
"""Run/plan the predeclared exact-H glass avoidability frontier.

The runner invokes the canonical collector independently at H in
``40,30,20,15,10,5``.  Each H has its own protocol hash, attempt ledger, and
artifact root.  A nonzero collector exit caused only by unmet acceptance quotas
is expected: the frontier needs every candidate outcome, not first success.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.glass_recovery_data import read_placement_manifest


FRONTIER_HORIZONS = (40, 30, 20, 15, 10, 5)
ORACLE_REJECTION_REASONS = {
    "no_oracle_recovery",
    "oracle_collision",
    "oracle_task_failure",
    "oracle_recapture_failure",
    "off_path_catastrophe",
    "off_path_timeout",
    "off_path_no_frames",
}


def _terminal_attempts(path: Path) -> dict[str, dict]:
    terminal = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") in {"accepted", "rejected", "failed"}:
            terminal[str(row["attempt_key"])] = row
    return terminal


def summarize_frontier_rows(
    rows: Iterable[Mapping],
    *,
    min_safe_task_success_rate: float,
) -> dict:
    rows = [dict(row) for row in rows]
    candidate_ids = sorted({str(row["placement_id"]) for row in rows})
    by_h: dict[int, list[dict]] = defaultdict(list)
    seen = set()
    for row in rows:
        key = (str(row["placement_id"]), int(row["horizon_actions"]))
        if key in seen:
            raise ValueError(f"duplicate frontier row {key}")
        seen.add(key)
        by_h[key[1]].append(row)
    missing = {
        horizon: sorted(set(candidate_ids) - {
            str(row["placement_id"]) for row in by_h.get(horizon, [])
        })
        for horizon in FRONTIER_HORIZONS
    }
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise ValueError(f"frontier does not cover every candidate/H: {missing}")

    horizon_rows = {}
    qualified = []
    for horizon in FRONTIER_HORIZONS:
        group = by_h[horizon]
        base_catastrophe = [row for row in group if row["base_catastrophe"] is True]
        safe = [row for row in base_catastrophe if row["oracle_safe_task_success"] is True]
        exact = [row for row in base_catastrophe if row["exact_h_replay_verified"] is True]
        rate = len(safe) / len(base_catastrophe) if base_catastrophe else 0.0
        exact_rate = len(exact) / len(base_catastrophe) if base_catastrophe else 0.0
        horizon_rows[str(horizon)] = {
            "candidates": len(group),
            "base_catastrophes": len(base_catastrophe),
            "oracle_safe_task_successes": len(safe),
            "oracle_safe_task_success_rate_given_base_catastrophe": rate,
            "exact_h_replay_rate_given_base_catastrophe": exact_rate,
            "rejection_reasons": dict(Counter(
                str(row.get("rejection_reason")) for row in group
                if row.get("rejection_reason")
            )),
        }
        if (
            base_catastrophe
            and rate >= min_safe_task_success_rate
            and exact_rate == 1.0
        ):
            qualified.append(horizon)
    preference = (20, 30, 40, 15, 10, 5)
    recommended = next((h for h in preference if h in qualified), None)
    return {
        "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "horizons": horizon_rows,
        "minimum_safe_task_success_rate": min_safe_task_success_rate,
        "qualified_horizons": qualified,
        "recommended_horizon_actions": recommended,
        "go": recommended is not None,
        "decision": (
            f"freeze H={recommended}" if recommended is not None
            else "no common tested H has usable exact-replay recoverability"
        ),
    }


def _collector_command(args: argparse.Namespace, horizon: int, counts: Mapping[str, int]) -> list[str]:
    output = Path(args.output_root) / f"h_{horizon}"
    return [
        sys.executable,
        "scripts/collect_glass_recovery_pairs.py",
        "--placements", str(Path(args.placements).resolve()),
        "--output", str(output.resolve()),
        "--checkpoint", args.checkpoint,
        "--checkpoint-revision", args.checkpoint_revision,
        "--rollout-seed", str(args.rollout_seed),
        "--unnorm-key", args.unnorm_key,
        "--max-train", str(counts["train"]),
        "--max-validation", str(counts["validation"]),
        "--max-heldout", str(counts["heldout"]),
        "--settle-steps", str(args.settle_steps),
        "--precrash-horizon", str(horizon),
        "--scan-steps", str(args.scan_steps),
        "--control-steps", str(args.control_steps),
        "--oracle-steps", str(args.oracle_steps),
        "--no-resume",
    ]


def _rows_for_horizon(root: Path, horizon: int) -> list[dict]:
    ledger = root / f"h_{horizon}" / "attempts.jsonl"
    if not ledger.is_file():
        raise RuntimeError(f"frontier collector did not write {ledger}")
    rows = []
    for event in _terminal_attempts(ledger).values():
        reason = event.get("reason")
        accepted = event.get("event") == "accepted"
        base_catastrophe = bool(
            accepted
            or (
                event.get("event") == "rejected"
                and reason not in {None, "no_base_crash"}
            )
        )
        pair_dir = root / f"h_{horizon}" / str(event.get("artifact_dir", ""))
        replay = None
        if (pair_dir / "nominal_replay.json").is_file():
            replay = json.loads((pair_dir / "nominal_replay.json").read_text())
        rows.append({
            "placement_id": event["placement_id"],
            "split": event.get("split"),
            "attempt_key": event["attempt_key"],
            "horizon_actions": horizon,
            "terminal_event": event.get("event"),
            "rejection_reason": reason,
            "base_catastrophe": base_catastrophe,
            "exact_h_replay_verified": bool(replay and replay.get("verified") is True),
            "oracle_safe_task_success": accepted,
            "oracle_catastrophe_or_failure": bool(
                reason in ORACLE_REJECTION_REASONS
            ),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--scan-steps", type=int, default=220)
    parser.add_argument("--control-steps", type=int, default=220)
    parser.add_argument("--oracle-steps", type=int, default=900)
    parser.add_argument("--min-candidates", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-safe-task-success-rate", type=float, default=0.5)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-commands", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    placements, _ = read_placement_manifest(args.placements)
    if not args.min_candidates <= len(placements) <= args.max_candidates:
        raise SystemExit(
            f"frontier needs {args.min_candidates}--{args.max_candidates} placements; "
            f"got {len(placements)}"
        )
    if not 0.0 <= args.min_safe_task_success_rate <= 1.0:
        raise SystemExit("minimum safe-task-success rate must lie in [0,1]")
    counts = Counter(placement.split for placement in placements)
    commands = [_collector_command(args, horizon, counts) for horizon in FRONTIER_HORIZONS]
    if args.print_commands:
        print("\n".join(" ".join(command) for command in commands))
        return
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for horizon, command in zip(FRONTIER_HORIZONS, commands):
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False)
        incomplete = root / f"h_{horizon}" / "collection_incomplete.json"
        if result.returncode != 0 and not incomplete.is_file():
            raise SystemExit(f"frontier H={horizon} failed outside expected quota miss")
    rows = [
        row for horizon in FRONTIER_HORIZONS for row in _rows_for_horizon(root, horizon)
    ]
    row_path = root / "frontier_rows.jsonl"
    row_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = {
        "schema_version": 1,
        "kind": "glass_avoidability_frontier",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "placements": str(Path(args.placements).resolve()),
        "checkpoint_revision": args.checkpoint_revision,
        "rollout_seed": args.rollout_seed,
        **summarize_frontier_rows(
            rows, min_safe_task_success_rate=args.min_safe_task_success_rate
        ),
    }
    (root / "frontier_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["go"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
