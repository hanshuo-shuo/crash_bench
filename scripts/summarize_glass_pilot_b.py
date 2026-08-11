#!/usr/bin/env python3
"""Summarize the frozen Pilot B frontier and task-0 collection gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from crashbench.glass_recovery_data import (
    read_placement_manifest,
    read_trajectory_manifest,
    validate_primary_pair,
)


SPLITS = ("train", "validation", "heldout")
OFFPATH_FAILURES = {
    "off_path_catastrophe", "off_path_timeout", "off_path_no_frames"
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terminal_attempts(path: Path) -> list[dict]:
    terminal: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") in {"accepted", "rejected", "failed"}:
            terminal[str(row["attempt_key"])] = row
    return list(terminal.values())


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(
    *, frontier_path: Path, placement_path: Path, dataset_root: Path
) -> dict:
    frontier = json.loads(frontier_path.read_text())
    placements, placement_payload = read_placement_manifest(placement_path)
    placements_by_id = {row.placement_id: row for row in placements}
    ledger_path = dataset_root / "attempts.jsonl"
    collection_path = dataset_root / "collection_summary.json"
    events = terminal_attempts(ledger_path)
    collection = json.loads(collection_path.read_text())

    accepted_ids = {
        str(row["placement_id"])
        for row in collection.get("accepted_verification", [])
        if row.get("primary_accepted") is True
    }
    accepted_groups = {}
    for split in SPLITS:
        records = read_trajectory_manifest(dataset_root / f"{split}.jsonl")
        by_pair: dict[str, list] = defaultdict(list)
        for record in records:
            by_pair[record.pair_id].append(record)
        for pair_id, group in by_pair.items():
            validate_primary_pair(group)
            accepted_groups[pair_id] = group
    individually_verified = set(accepted_groups)

    scientific_events = [row for row in events if row.get("event") != "failed"]
    base_catastrophes = [
        row for row in scientific_events
        if row.get("event") == "accepted" or row.get("reason") != "no_base_crash"
    ]
    exact_replays = []
    offpath_successes = []
    offpath_failures = []
    for row in base_catastrophes:
        pair_root = dataset_root / str(row.get("artifact_dir", ""))
        replay_path = pair_root / "nominal_replay.json"
        if replay_path.is_file() and json.loads(replay_path.read_text()).get("verified") is True:
            exact_replays.append(row)
        if (pair_root / "offpath_probe.json").is_file():
            offpath_successes.append(row)
        elif row.get("reason") in OFFPATH_FAILURES:
            offpath_failures.append(row)

    offpath_attempts = offpath_successes + offpath_failures
    accepted_counts = Counter(
        placements_by_id[pair_id].split for pair_id in accepted_ids
    )
    accepted_sources = {
        split: {
            placements_by_id[pair_id].source_state_sha256
            for pair_id in accepted_ids
            if placements_by_id[pair_id].split == split
        }
        for split in SPLITS
    }
    accepted_families = Counter(
        str(placements_by_id[pair_id].metadata["geometry_family"])
        for pair_id in accepted_ids
    )
    declared_families = {
        str(row.metadata["geometry_family"]) for row in placements
    }
    terminal_placement_counts = Counter(str(row["placement_id"]) for row in events)
    primary = collection["metadata"]["primary_protocol"]
    horizon = int(primary["precrash_horizon_actions"])

    metrics = {
        "path_proposal_base_catastrophe_yield": ratio(
            len(base_catastrophes), len(scientific_events)
        ),
        "exact_nominal_replay_rate_given_base_catastrophe": ratio(
            len(exact_replays), len(base_catastrophes)
        ),
        "accepted_oracle_independent_verification_rate": ratio(
            len(individually_verified), len(accepted_ids)
        ),
        "oracle_safe_task_success_yield_given_base_catastrophe": ratio(
            len(accepted_ids), len(base_catastrophes)
        ),
        "matched_offpath_catastrophe_rate": ratio(
            sum(row.get("reason") == "off_path_catastrophe" for row in offpath_failures),
            len(offpath_attempts),
        ),
        "matched_offpath_task_success_rate": ratio(
            len(offpath_successes), len(offpath_attempts)
        ),
    }
    checks = {
        "frontier_selected_fixed_h": (
            frontier.get("go") is True
            and int(frontier.get("recommended_horizon_actions", -1)) == horizon
        ),
        "path_proposal_base_catastrophe_yield_at_least_30_percent": (
            metrics["path_proposal_base_catastrophe_yield"] >= 0.30
        ),
        "exact_nominal_replay_100_percent": (
            metrics["exact_nominal_replay_rate_given_base_catastrophe"] == 1.0
        ),
        "accepted_oracle_recapture_at_least_90_percent": (
            metrics["accepted_oracle_independent_verification_rate"] >= 0.90
            and individually_verified == accepted_ids
        ),
        "matched_offpath_catastrophe_0_percent": (
            bool(offpath_attempts)
            and metrics["matched_offpath_catastrophe_rate"] == 0.0
        ),
        "matched_offpath_task_success_at_least_80_percent": (
            metrics["matched_offpath_task_success_rate"] >= 0.80
        ),
        "accepted_pair_counts_10_5_5": all(
            accepted_counts[split] >= required
            for split, required in {"train": 10, "validation": 5, "heldout": 5}.items()
        ),
        "accepted_source_state_counts_8_5_5": all(
            len(accepted_sources[split]) >= required
            for split, required in {"train": 8, "validation": 5, "heldout": 5}.items()
        ),
        "validation_nonempty": accepted_counts["validation"] > 0,
        "accepted_not_concentrated_in_one_family": len(accepted_families) > 1,
        "each_declared_family_represented": (
            set(accepted_families) == declared_families
        ),
        "one_terminal_attempt_per_candidate": all(
            count == 1 for count in terminal_placement_counts.values()
        ),
        "no_technical_failures": not any(
            row.get("event") == "failed" for row in events
        ),
    }
    no_go_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "kind": "glass_recovery_pilot_b_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pilot_b_go" if not no_go_reasons else "pilot_b_no_go",
        "go": not no_go_reasons,
        "pilot_c_allowed": not no_go_reasons,
        "no_go_reasons": no_go_reasons,
        "horizon_actions": horizon,
        "frontier": {
            "path": str(frontier_path.resolve()),
            "sha256": file_sha256(frontier_path),
            "go": frontier.get("go"),
            "recommended_horizon_actions": frontier.get(
                "recommended_horizon_actions"
            ),
            "horizons": frontier.get("horizons"),
        },
        "inputs": {
            "placement_manifest": str(placement_path.resolve()),
            "placement_manifest_sha256": file_sha256(placement_path),
            "placement_design": placement_payload.get("design"),
            "dataset_root": str(dataset_root.resolve()),
            "attempt_ledger_sha256": file_sha256(ledger_path),
            "collection_summary_sha256": file_sha256(collection_path),
        },
        "attempt_accounting": {
            "terminal_attempts": len(events),
            "scientific_terminal_attempts": len(scientific_events),
            "technical_failures": sum(row.get("event") == "failed" for row in events),
            "base_catastrophes": len(base_catastrophes),
            "exact_replays": len(exact_replays),
            "offpath_attempts": len(offpath_attempts),
            "offpath_task_successes": len(offpath_successes),
            "accepted_pairs": len(accepted_ids),
            "terminal_events": dict(Counter(str(row.get("event")) for row in events)),
            "rejection_reasons": dict(Counter(
                str(row.get("reason")) for row in events if row.get("reason")
            )),
        },
        "metrics": metrics,
        "accepted_cohort": {
            "pair_counts_by_split": {
                split: accepted_counts[split] for split in SPLITS
            },
            "source_state_counts_by_split": {
                split: len(accepted_sources[split]) for split in SPLITS
            },
            "family_counts": dict(accepted_families),
            "declared_families": sorted(declared_families),
            "pair_ids": sorted(accepted_ids),
        },
        "go_checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-summary", required=True)
    parser.add_argument("--placements", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    payload = summarize(
        frontier_path=Path(args.frontier_summary),
        placement_path=Path(args.placements),
        dataset_root=Path(args.dataset),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["go"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
