#!/usr/bin/env python3
"""Author the P1 timing-and-option-choice benchmark from exact-state branches.

The benchmark unit is a trajectory group (source state, placement, condition),
with matched T-30/T-20/T-10/T-5 anchors.  The authoring output keeps the four
decision classes explicit instead of hiding them behind one prevalence-weighted
overall score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from crashbench.counterfactual_router import OPTIONS
from crashbench.glass_recovery_data import canonical_sha256
from scripts.train_minimal_counterfactual_router import load_capture


P1_HORIZONS = (30, 20, 10, 5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_signature(outcomes: Sequence[str]) -> str:
    base, detour, retreat = map(str, outcomes)
    if (
        base == "catastrophe"
        and detour == "task_success"
        and retreat == "safe_noncompletion"
    ):
        return "recoverable"
    if (
        base == "catastrophe"
        and detour != "task_success"
        and retreat == "safe_noncompletion"
    ):
        return "loss_control"
    if (
        base == "task_success"
        and detour != "task_success"
        and retreat == "safe_noncompletion"
    ):
        return "unnecessary"
    return "other"


def correct_option(signature: str) -> str | None:
    return {
        "recoverable": "detour_complete",
        "loss_control": "retreat_hold",
        "unnecessary": "base_continue",
        "library_unsolved": "retreat_hold",
    }.get(signature)


def _trajectory_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["source_state_sha256"]),
        str(row["placement_id"]),
        str(row["condition"]),
    )


def build_timing_records(
    metadata: Sequence[Mapping[str, Any]],
    outcomes: np.ndarray,
    *,
    horizons: Sequence[int] = P1_HORIZONS,
) -> dict[str, Any]:
    horizon_set = set(map(int, horizons))
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(metadata):
        horizon = int(row["horizon_actions"])
        if horizon not in horizon_set:
            continue
        signature = point_signature(outcomes[index])
        record = {
            "decision_id": str(row["decision_id"]),
            "source_state_sha256": str(row["source_state_sha256"]),
            "placement_id": str(row["placement_id"]),
            "placement_key": row.get("placement_key"),
            "split": str(row["split"]),
            "condition": str(row["condition"]),
            "horizon_actions": horizon,
            "outcomes": {
                option: str(outcomes[index, option_index])
                for option_index, option in enumerate(OPTIONS)
            },
            "point_signature": signature,
            "correct_option": correct_option(signature),
        }
        groups[_trajectory_key(row)].append(record)

    trajectories = []
    benchmark_records = []
    for key, points in sorted(groups.items()):
        points.sort(key=lambda row: -int(row["horizon_actions"]))
        by_horizon = {int(row["horizon_actions"]): row for row in points}
        complete = set(by_horizon) == horizon_set
        recoverable = [
            row for row in points if row["point_signature"] == "recoverable"
        ]
        loss_control = [
            row for row in points if row["point_signature"] == "loss_control"
        ]
        transitions = [
            (early, late)
            for early in recoverable
            for late in loss_control
            if int(early["horizon_actions"]) > int(late["horizon_actions"])
        ]
        timing_pair = None
        if transitions:
            early, late = max(
                transitions,
                key=lambda pair: (
                    int(pair[0]["horizon_actions"]) - int(pair[1]["horizon_actions"]),
                    int(pair[0]["horizon_actions"]),
                ),
            )
            pair_id = canonical_sha256({
                "trajectory": key,
                "early_decision_id": early["decision_id"],
                "late_decision_id": late["decision_id"],
            })
            early_record = {
                **early,
                "benchmark_class": "early_recoverable",
                "correct_option": "detour_complete",
                "timing_pair_id": pair_id,
            }
            late_record = {
                **late,
                "benchmark_class": "late_loss_control",
                "correct_option": "retreat_hold",
                "timing_pair_id": pair_id,
            }
            benchmark_records.extend((early_record, late_record))
            timing_pair = {
                "timing_pair_id": pair_id,
                "early_horizon_actions": int(early["horizon_actions"]),
                "late_horizon_actions": int(late["horizon_actions"]),
                "early_decision_id": early["decision_id"],
                "late_decision_id": late["decision_id"],
            }

        library_unsolved = bool(
            complete
            and all(
                row["outcomes"]["base_continue"] == "catastrophe"
                and row["outcomes"]["detour_complete"] != "task_success"
                and row["outcomes"]["retreat_hold"] == "safe_noncompletion"
                for row in points
            )
        )
        if library_unsolved:
            # Use the latest valid anchor: this is the strongest fail-safe case.
            selected = min(points, key=lambda row: int(row["horizon_actions"]))
            benchmark_records.append({
                **selected,
                "benchmark_class": "library_unsolved",
                "correct_option": "retreat_hold",
                "timing_pair_id": None,
            })

        unnecessary = [
            row for row in points if row["point_signature"] == "unnecessary"
        ]
        if unnecessary:
            # One point per trajectory prevents long/easy trajectories dominating.
            selected = min(
                unnecessary,
                key=lambda row: (
                    abs(int(row["horizon_actions"]) - 20),
                    -int(row["horizon_actions"]),
                ),
            )
            benchmark_records.append({
                **selected,
                "benchmark_class": "unnecessary",
                "correct_option": "base_continue",
                "timing_pair_id": None,
            })

        trajectories.append({
            "trajectory_id": canonical_sha256({"trajectory": key}),
            "source_state_sha256": key[0],
            "placement_id": key[1],
            "condition": key[2],
            "split": points[0]["split"],
            "complete_horizon_grid": complete,
            "horizons_present": sorted(by_horizon, reverse=True),
            "point_signatures": {
                str(horizon): by_horizon[horizon]["point_signature"]
                for horizon in sorted(by_horizon, reverse=True)
            },
            "timing_pair": timing_pair,
            "library_unsolved": library_unsolved,
            "has_unnecessary_state": bool(unnecessary),
        })

    # A decision can be useful to only one benchmark class in the primary table.
    priority = {
        "early_recoverable": 0,
        "late_loss_control": 0,
        "library_unsolved": 1,
        "unnecessary": 2,
    }
    unique_records = {}
    for row in benchmark_records:
        decision_id = row["decision_id"]
        previous = unique_records.get(decision_id)
        if previous is None or priority[row["benchmark_class"]] < priority[
            previous["benchmark_class"]
        ]:
            unique_records[decision_id] = row
    benchmark_records = sorted(
        unique_records.values(),
        key=lambda row: (
            row["split"], row["benchmark_class"], row["source_state_sha256"],
            row["placement_id"], -int(row["horizon_actions"]),
        ),
    )
    return {"trajectories": trajectories, "records": benchmark_records}


def _counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["benchmark_class"]) for row in records)
    return {
        name: int(counts[name])
        for name in (
            "early_recoverable", "late_loss_control",
            "unnecessary", "library_unsolved",
        )
    }


def author(capture: Path, output: Path, horizons: Sequence[int]) -> dict[str, Any]:
    data = load_capture(capture, catastrophe_cost=1.0)
    built = build_timing_records(
        data["metadata"], data["outcomes"], horizons=horizons,
    )
    records = built["records"]
    trajectories = built["trajectories"]
    complete = [row for row in trajectories if row["complete_horizon_grid"]]
    timing_pairs = [row for row in trajectories if row["timing_pair"] is not None]
    class_counts = _counts(records)
    split_counts = {
        split: _counts([row for row in records if row["split"] == split])
        for split in ("train", "calibration", "development")
    }
    source_counts = {
        split: len({
            row["source_state_sha256"] for row in records if row["split"] == split
        })
        for split in ("train", "calibration", "development")
    }
    missing_classes = [name for name, count in class_counts.items() if count == 0]
    result = {
        "schema_version": 1,
        "kind": "p1_timing_option_choice_benchmark_authoring",
        "status": "development_discovery_not_confirmatory",
        "source_capture": str(capture),
        "source_artifact_sha256": {
            name: _sha256(capture / name)
            for name in (
                "capture_manifest.json", "decision_metadata.json",
                "option_rollouts.jsonl", "decision_features.npz",
            )
        },
        "protocol": {
            "trajectory_unit": "source_state_sha256 + placement_id + condition",
            "horizons": list(map(int, horizons)),
            "options": list(OPTIONS),
            "class_macro_average_required": True,
            "timing_pair_rule": (
                "same trajectory has an earlier recoverable anchor and a later "
                "loss-control anchor"
            ),
            "tie_break": (
                "RetreatHold is the fail-safe choice when no option completes and "
                "Detour is catastrophe or safe noncompletion"
            ),
            "selection_unit": "at most one unnecessary and one library-unsolved point per trajectory",
        },
        "coverage": {
            "trajectory_groups": len(trajectories),
            "complete_horizon_groups": len(complete),
            "timing_choice_pairs": len(timing_pairs),
            "benchmark_records": len(records),
            "class_counts": class_counts,
            "class_counts_by_split": split_counts,
            "source_counts_by_split": source_counts,
            "missing_classes": missing_classes,
        },
        "go_for_fresh_pilot": bool(
            not missing_classes and len(timing_pairs) >= 1
        ),
        "records": records,
        "trajectories": trajectories,
        "fresh_protocol_requirements": {
            "source_disjoint_from": sorted(set(map(str, data["sources"]))),
            "capture_all_horizons_in_one_job": True,
            "do_not_select_one_anchor_after_outcomes": True,
            "retain_raw_fresh_features": True,
            "report": [
                "four-class macro accuracy",
                "early-vs-late paired option accuracy",
                "success/catastrophe/safe-noncompletion by class",
                "first-crossing trigger timing in the later dynamic phase",
            ],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    fieldnames = [
        "benchmark_class", "correct_option", "split", "condition",
        "horizon_actions", "source_state_sha256", "placement_id",
        "decision_id", "timing_pair_id", "base_outcome", "detour_outcome",
        "retreat_outcome",
    ]
    with (output / "benchmark_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow({
                **{key: row.get(key) for key in fieldnames[:-3]},
                "base_outcome": row["outcomes"]["base_continue"],
                "detour_outcome": row["outcomes"]["detour_complete"],
                "retreat_outcome": row["outcomes"]["retreat_hold"],
            })
    lines = [
        "# P1 timing-and-option-choice benchmark authoring",
        "",
        "Status: **development discovery; not confirmatory**.",
        "",
        f"Source capture: `{capture}`",
        "",
        "| Class | Count | Correct option |",
        "|---|---:|---|",
        f"| early recoverable | {class_counts['early_recoverable']} | Detour |",
        f"| late loss control | {class_counts['late_loss_control']} | Retreat |",
        f"| unnecessary | {class_counts['unnecessary']} | Base |",
        f"| library unsolved | {class_counts['library_unsolved']} | Retreat/fail-safe |",
        "",
        f"Complete four-horizon groups: **{len(complete)}**; clean same-trajectory "
        f"timing pairs: **{len(timing_pairs)}**.",
        "",
        "The old capture is sufficient to freeze the schema, but its timing pairs "
        "are discovery evidence only. The next pilot must use new source states and "
        "capture all four anchors together.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(P1_HORIZONS)
    )
    args = parser.parse_args()
    result = author(
        args.capture.resolve(), args.output.resolve(), args.horizons,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "coverage": result["coverage"],
        "go_for_fresh_pilot": result["go_for_fresh_pilot"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
