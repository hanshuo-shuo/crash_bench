#!/usr/bin/env python3
"""Author dense P3 recovery-window labels from exact-state option branches.

The raw collector retains its historical ``retreat_hold`` slot for compatibility
with the frozen Router.  P3 requires that slot to have been executed with
``--retreat-mode hold``; every authored record therefore names the option
``FailSafeHold`` and rejects directional Retreat captures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from crashbench.counterfactual_router import (
    FrozenOutcomeRouter,
    OPTIONS,
    calibrated_advantage_lcb,
)
from crashbench.glass_recovery_data import canonical_sha256
from scripts.train_minimal_counterfactual_router import load_capture


DENSE_HORIZONS = tuple(range(30, 1, -2)) + (1,)
DISPLAY_OPTIONS = ("Base", "Detour", "FailSafeHold")
OUTCOME_RANK = {"catastrophe": 0, "safe_noncompletion": 1, "task_success": 2}
MISSED_RECOVERABLE_PLACEMENTS = (
    "glass_recovery_heldout_0010",
    "glass_recovery_heldout_0019",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def label_anchor(base: str, detour: str, hold: str) -> dict[str, Any]:
    """Apply the predeclared P3 option ordering and fail-safe rules."""

    outcomes = (str(base), str(detour), str(hold))
    unknown = set(outcomes) - set(OUTCOME_RANK)
    if unknown:
        raise ValueError(f"unknown option outcomes: {sorted(unknown)}")

    recovery_open = base == "catastrophe" and detour == "task_success"
    loss_control = (
        base == "catastrophe"
        and detour != "task_success"
        and hold == "safe_noncompletion"
    )
    violation = hold == "catastrophe"

    # Base wins a Base/Detour success tie.  If nothing completes, a safe Hold
    # wins exactly as required by the fail-safe contract.  Remaining cases use
    # outcome rank, then Base > Detour > Hold to avoid ineffective takeover.
    if base == "task_success":
        preferred = "Base"
    elif detour == "task_success":
        preferred = "Detour"
    elif hold == "task_success":
        preferred = "FailSafeHold"
    elif hold == "safe_noncompletion":
        preferred = "FailSafeHold"
    else:
        best = max(OUTCOME_RANK[value] for value in outcomes)
        preferred = next(
            option for option, value in zip(DISPLAY_OPTIONS, outcomes)
            if OUTCOME_RANK[value] == best
        )

    return {
        "base_outcome": base,
        "detour_outcome": detour,
        "hold_outcome": hold,
        "preferred_option": preferred,
        "intervention_needed": bool(
            base != "task_success"
            and (base == "catastrophe" or preferred != "Base")
        ),
        "recovery_open": bool(recovery_open),
        "loss_control": bool(loss_control),
        "fail_safe_contract_violation": bool(violation),
        "task_completion_unsolved": not any(
            value == "task_success" for value in outcomes
        ),
    }


def recovery_intervals(
    records: Sequence[Mapping[str, Any]],
    *,
    horizon_grid: Sequence[int] = DENSE_HORIZONS,
) -> list[dict[str, Any]]:
    """Return every sampled contiguous recovery interval without monotonicity assumptions."""

    order = {int(horizon): index for index, horizon in enumerate(horizon_grid)}
    positive = sorted(
        [row for row in records if bool(row["recovery_open"])],
        key=lambda row: order[int(row["horizon_actions"])],
    )
    intervals: list[list[Mapping[str, Any]]] = []
    for row in positive:
        if (
            not intervals
            or order[int(row["horizon_actions"])]
            != order[int(intervals[-1][-1]["horizon_actions"])] + 1
        ):
            intervals.append([row])
        else:
            intervals[-1].append(row)

    result = []
    for index, interval in enumerate(intervals):
        earliest = int(interval[0]["horizon_actions"])
        latest = int(interval[-1]["horizon_actions"])
        result.append({
            "interval_index": index,
            "earliest_recoverable_anchor": earliest,
            "latest_recoverable_anchor": latest,
            "recovery_window_span_actions": earliest - latest,
            "recovery_anchor_count": len(interval),
            "horizons": [int(row["horizon_actions"]) for row in interval],
        })
    return result


def _annotate_window_closure(
    records: list[dict[str, Any]], intervals: Sequence[Mapping[str, Any]]
) -> None:
    by_horizon = {}
    for interval in intervals:
        latest = int(interval["latest_recoverable_anchor"])
        for horizon in interval["horizons"]:
            by_horizon[int(horizon)] = (
                int(interval["interval_index"]), latest, int(horizon) - latest
            )
    for row in records:
        values = by_horizon.get(int(row["horizon_actions"]))
        row["recovery_interval_index"] = None if values is None else values[0]
        row["window_closure_horizon_actions"] = None if values is None else values[1]
        row["actions_to_window_closure"] = None if values is None else values[2]


def _transition_sequence(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    previous = None
    for row in records:
        option = str(row["preferred_option"])
        if previous is None or option != previous["preferred_option"]:
            transitions.append({
                "horizon_actions": int(row["horizon_actions"]),
                "preferred_option": option,
            })
        previous = row
    return transitions


def _router_point(router: FrozenOutcomeRouter, catastrophe_cost: float, target: float) -> dict[str, Any]:
    frontier = router.manifest["calibration"]["router_frontier"]
    point = frontier[f"lambda_{catastrophe_cost:g}"][f"target_{target:.1f}"]
    return {"catastrophe_cost": float(catastrophe_cost), "target": float(target), **point}


def _score_dense_rows(
    data: Mapping[str, Any],
    router: FrozenOutcomeRouter,
    *,
    catastrophe_cost: float,
    margin: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    output_vectors: dict[str, np.ndarray] = {}
    arrays = data["arrays"]
    for index, metadata in enumerate(data["metadata"]):
        mask = np.asarray(arrays["history_mask"][index])
        valid = np.flatnonzero(mask > 0)
        if not len(valid):
            raise ValueError(f"decision {metadata['decision_id']} has no valid history")
        frame = int(valid[-1])
        prediction = router.predict(
            arrays["hidden"][index, frame],
            arrays["robot_state"][index, frame],
            arrays["nominal_action"][index, frame],
        )
        probabilities = np.asarray(
            prediction["option_outcome_probabilities"], dtype=np.float64
        )
        utility, advantage, lcb = calibrated_advantage_lcb(
            probabilities,
            catastrophe_cost=catastrophe_cost,
            intervention_margin=margin,
        )
        candidate = 1 + int(np.argmax(advantage[1:]))
        outcome = {
            option: str(data["outcomes"][index, option_index])
            for option_index, option in enumerate(OPTIONS)
        }
        labels = label_anchor(
            outcome["base_continue"],
            outcome["detour_complete"],
            outcome["retreat_hold"],
        )
        decision_id = str(metadata["decision_id"])
        output_vector = np.concatenate((
            probabilities.reshape(-1),
            [float(prediction["base_catastrophe_probability"])],
        ))
        output_vectors[decision_id] = output_vector
        rows.append({
            "schema_version": 1,
            "record_type": "dense_glass_anchor",
            "decision_id": decision_id,
            "trajectory_id": canonical_sha256({
                "source_state_sha256": metadata["source_state_sha256"],
                "placement_id": metadata["placement_id"],
                "condition": metadata["condition"],
            }),
            "placement_id": str(metadata["placement_id"]),
            "placement_key": metadata.get("placement_key"),
            "source_state_sha256": str(metadata["source_state_sha256"]),
            "split": str(metadata["split"]),
            "condition": str(metadata["condition"]),
            "horizon_actions": int(metadata["horizon_actions"]),
            "action_index": int(metadata["matched_scan_index"]),
            "feature_index": int(metadata["feature_index"]),
            "branch_start_hashes": dict(metadata["branch_start_hashes"]),
            **labels,
            "hard_negative": False,
            "old_router_advantage": float(np.max(advantage[1:])),
            "old_router_detour_advantage": float(advantage[1]),
            "old_router_hold_advantage": float(advantage[2]),
            "old_router_candidate_option": DISPLAY_OPTIONS[candidate],
            "old_router_pointwise_lcb": float(np.max(lcb[1:])),
            "old_router_utility": utility.tolist(),
            "old_router_option_outcome_probabilities": probabilities.tolist(),
            "old_router_base_catastrophe_probability": float(
                prediction["base_catastrophe_probability"]
            ),
            "router_output_feature": output_vector.tolist(),
        })
    return rows, output_vectors


def _true_regions(actions: Sequence[int], values: Sequence[float], threshold: float) -> list[list[int]]:
    regions: list[list[int]] = []
    current: list[int] = []
    previous = None
    for index, (action, value) in enumerate(zip(actions, values)):
        if float(value) > float(threshold):
            if previous is None or int(action) != previous + 1:
                if current:
                    regions.append(current)
                current = []
            current.append(index)
            previous = int(action)
        elif current:
            regions.append(current)
            current = []
            previous = None
    if current:
        regions.append(current)
    return regions


def extract_hard_control_records(
    episodes: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    pointwise_margin: float,
    top_by_peak: int = 2,
    top_by_duration: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Select the strongest and longest old-Router regions on Base-success controls."""

    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        by_episode[str(row["episode_id"])].append(row)
    controls = {
        str(row["episode_id"]): row for row in episodes
        if row["condition"] in {"offpath", "noglass"}
        and row["reference_base_outcome"] == "task_success"
    }
    regions = []
    for episode_id, episode in controls.items():
        trace = sorted(by_episode.get(episode_id, []), key=lambda row: int(row["action_index"]))
        if not trace:
            raise ValueError(f"hard-control episode {episode_id} lacks a Router trace")
        actions = [int(row["action_index"]) for row in trace]
        values = [float(max(row["advantage_vs_base"][1:])) for row in trace]
        for indices in _true_regions(actions, values, pointwise_margin):
            selected = [trace[index] for index in indices]
            region_id = canonical_sha256({
                "episode_id": episode_id,
                "start": int(selected[0]["action_index"]),
                "end": int(selected[-1]["action_index"]),
                "pointwise_margin": float(pointwise_margin),
            })
            regions.append({
                "region_id": region_id,
                "episode_id": episode_id,
                "placement_id": str(episode["placement_id"]),
                "source_state_sha256": str(episode["source_state_sha256"]),
                "condition": str(episode["condition"]),
                "start_action_index": int(selected[0]["action_index"]),
                "end_action_index": int(selected[-1]["action_index"]),
                "duration_actions": len(selected),
                "peak_advantage": max(values[index] for index in indices),
                "trace_rows": selected,
            })
    if not regions:
        raise ValueError("no Base-success control region exceeds the pointwise margin")

    by_peak = sorted(
        regions, key=lambda row: (-row["peak_advantage"], -row["duration_actions"], row["region_id"])
    )
    by_duration = sorted(
        regions, key=lambda row: (-row["duration_actions"], -row["peak_advantage"], row["region_id"])
    )
    peak_rank = {row["region_id"]: rank for rank, row in enumerate(by_peak, 1)}
    duration_rank = {row["region_id"]: rank for rank, row in enumerate(by_duration, 1)}
    selected_ids = {
        row["region_id"] for row in by_peak[:max(0, int(top_by_peak))]
    } | {
        row["region_id"] for row in by_duration[:max(0, int(top_by_duration))]
    }
    selected_regions = []
    records = []
    vectors: dict[str, np.ndarray] = {}
    for region in regions:
        if region["region_id"] not in selected_ids:
            continue
        reasons = []
        if peak_rank[region["region_id"]] <= top_by_peak:
            reasons.append("top_peak_advantage")
        if duration_rank[region["region_id"]] <= top_by_duration:
            reasons.append("top_duration")
        selected_regions.append({
            **{key: value for key, value in region.items() if key != "trace_rows"},
            "peak_rank": peak_rank[region["region_id"]],
            "duration_rank": duration_rank[region["region_id"]],
            "selection_reasons": reasons,
        })
        for trace in region["trace_rows"]:
            probabilities = np.asarray(
                trace["option_outcome_probabilities"], dtype=np.float64
            )
            advantage = np.asarray(trace["advantage_vs_base"], dtype=np.float64)
            candidate = 1 + int(np.argmax(advantage[1:]))
            record_id = canonical_sha256({
                "record_type": "hard_control_negative",
                "episode_id": region["episode_id"],
                "action_index": int(trace["action_index"]),
            })
            vector = np.concatenate((
                probabilities.reshape(-1),
                [float(trace["base_catastrophe_probability"])],
            ))
            vectors[record_id] = vector
            records.append({
                "schema_version": 1,
                "record_type": "hard_control_negative",
                "decision_id": record_id,
                "trajectory_id": str(region["episode_id"]),
                "placement_id": region["placement_id"],
                "placement_key": None,
                "source_state_sha256": region["source_state_sha256"],
                "split": "development",
                "condition": region["condition"],
                "horizon_actions": None,
                "action_index": int(trace["action_index"]),
                "feature_index": None,
                "branch_start_hashes": None,
                "base_outcome": "task_success",
                "detour_outcome": None,
                "hold_outcome": None,
                "preferred_option": "Base",
                "intervention_needed": False,
                "recovery_open": False,
                "loss_control": False,
                "fail_safe_contract_violation": False,
                "task_completion_unsolved": False,
                "hard_negative": True,
                "recovery_interval_index": None,
                "window_closure_horizon_actions": None,
                "actions_to_window_closure": None,
                "hard_control_region_id": region["region_id"],
                "hard_control_region_duration_actions": region["duration_actions"],
                "hard_control_region_peak_advantage": region["peak_advantage"],
                "hard_control_region_peak_rank": peak_rank[region["region_id"]],
                "hard_control_region_duration_rank": duration_rank[region["region_id"]],
                "hard_control_selection_reasons": reasons,
                "old_router_advantage": float(np.max(advantage[1:])),
                "old_router_detour_advantage": float(advantage[1]),
                "old_router_hold_advantage": float(advantage[2]),
                "old_router_candidate_option": DISPLAY_OPTIONS[candidate],
                "old_router_pointwise_lcb": float(np.max(advantage[1:]) - pointwise_margin),
                "old_router_utility": trace.get("utility"),
                "old_router_option_outcome_probabilities": probabilities.tolist(),
                "old_router_base_catastrophe_probability": float(
                    trace["base_catastrophe_probability"]
                ),
                "router_output_feature": vector.tolist(),
            })
    selected_regions.sort(key=lambda row: (row["peak_rank"], row["duration_rank"]))
    records.sort(key=lambda row: (row["placement_id"], row["condition"], row["action_index"]))
    return records, selected_regions, vectors


def feature_overlap_summary(
    recovery_vectors: Sequence[np.ndarray], hard_vectors: Sequence[np.ndarray]
) -> dict[str, Any]:
    """Quantify overlap in the frozen Router's 10-D prediction-output space."""

    recovery = np.asarray(recovery_vectors, dtype=np.float64)
    hard = np.asarray(hard_vectors, dtype=np.float64)
    if not len(recovery) or not len(hard):
        return {"available": False, "reason": "one comparison class is empty"}
    pooled = np.vstack((recovery, hard))
    scale = pooled.std(axis=0)
    scale[scale < 1e-8] = 1.0
    recovery = (recovery - pooled.mean(axis=0)) / scale
    hard = (hard - pooled.mean(axis=0)) / scale
    cross = np.linalg.norm(recovery[:, None, :] - hard[None, :, :], axis=2)
    recovery_cross = cross.min(axis=1)
    hard_cross = cross.min(axis=0)

    within_recovery = None
    hard_closer_fraction = None
    if len(recovery) > 1:
        distances = np.linalg.norm(
            recovery[:, None, :] - recovery[None, :, :], axis=2
        )
        np.fill_diagonal(distances, np.inf)
        within_recovery = distances.min(axis=1)
        hard_closer_fraction = float(np.mean(recovery_cross <= within_recovery))

    all_points = np.vstack((recovery, hard))
    labels = np.asarray([1] * len(recovery) + [0] * len(hard))
    distances = np.linalg.norm(
        all_points[:, None, :] - all_points[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    prediction = labels[np.argmin(distances, axis=1)]
    recovery_accuracy = float(np.mean(prediction[:len(recovery)] == 1))
    hard_accuracy = float(np.mean(prediction[len(recovery):] == 0))
    return {
        "available": True,
        "space": (
            "standardized frozen-Router output vector: 3 options x 3 outcome "
            "probabilities plus Base catastrophe probability"
        ),
        "dimensions": int(pooled.shape[1]),
        "recovery_open_points": int(len(recovery)),
        "hard_negative_points": int(len(hard)),
        "median_recovery_to_hard_distance": float(np.median(recovery_cross)),
        "median_hard_to_recovery_distance": float(np.median(hard_cross)),
        "median_recovery_to_recovery_distance": (
            None if within_recovery is None else float(np.median(within_recovery))
        ),
        "recovery_points_with_hard_neighbor_no_farther_than_recovery_neighbor_fraction": (
            hard_closer_fraction
        ),
        "leave_one_out_1nn_recovery_accuracy": recovery_accuracy,
        "leave_one_out_1nn_hard_negative_accuracy": hard_accuracy,
        "leave_one_out_1nn_balanced_accuracy": float(
            (recovery_accuracy + hard_accuracy) / 2.0
        ),
    }


def _trajectory_summaries(
    records: Sequence[dict[str, Any]], horizon_grid: Sequence[int]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["record_type"] == "dense_glass_anchor":
            groups[str(row["trajectory_id"])].append(row)
    summaries = []
    expected = list(map(int, horizon_grid))
    order = {horizon: index for index, horizon in enumerate(expected)}
    for trajectory_id, points in groups.items():
        points.sort(key=lambda row: order[int(row["horizon_actions"])])
        present = [int(row["horizon_actions"]) for row in points]
        intervals = recovery_intervals(points, horizon_grid=expected)
        _annotate_window_closure(points, intervals)
        signatures = [
            "recovery_open" if row["recovery_open"]
            else "loss_control" if row["loss_control"]
            else "base_success" if row["base_outcome"] == "task_success"
            else "other"
            for row in points
        ]
        nonmonotonic = any(
            signatures[i] == "recovery_open"
            and "recovery_open" in signatures[i + 2:]
            and signatures[i + 1] != "recovery_open"
            for i in range(len(signatures) - 2)
        )
        transitions = _transition_sequence(points)
        summaries.append({
            "trajectory_id": trajectory_id,
            "placement_id": points[0]["placement_id"],
            "source_state_sha256": points[0]["source_state_sha256"],
            "condition": points[0]["condition"],
            "horizons_present": present,
            "missing_horizons": [h for h in expected if h not in set(present)],
            "complete_dense_grid": present == expected,
            "recovery_open_horizons": [
                int(row["horizon_actions"]) for row in points if row["recovery_open"]
            ],
            "recovery_intervals": intervals,
            "earliest_recoverable_anchor": (
                None if not intervals else max(
                    int(interval["earliest_recoverable_anchor"]) for interval in intervals
                )
            ),
            "latest_recoverable_anchor": (
                None if not intervals else min(
                    int(interval["latest_recoverable_anchor"]) for interval in intervals
                )
            ),
            "recovery_window_span_actions": (
                None if not intervals else max(
                    int(interval["earliest_recoverable_anchor"]) for interval in intervals
                ) - min(
                    int(interval["latest_recoverable_anchor"]) for interval in intervals
                )
            ),
            "recovery_anchor_count": sum(
                int(interval["recovery_anchor_count"]) for interval in intervals
            ),
            "recovery_windows_contiguous": len(intervals) <= 1,
            "nonmonotonic_recovery": bool(nonmonotonic),
            "has_recoverable_to_loss_control_transition": any(
                bool(points[early_index]["recovery_open"])
                and any(
                    bool(later["loss_control"])
                    for later in points[early_index + 1:]
                )
                for early_index in range(len(points))
            ),
            "preferred_option_transitions": transitions,
            "preferred_option_transition_string": " → ".join(
                transition["preferred_option"] for transition in transitions
            ),
            "library_unsolved": bool(
                any(row["base_outcome"] == "catastrophe" for row in points)
                and not any(row["recovery_open"] for row in points)
            ),
            "fail_safe_contract_violations": sum(
                bool(row["fail_safe_contract_violation"]) for row in points
            ),
            "records": points,
        })
    return sorted(summaries, key=lambda row: row["placement_id"])


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    preferred = [
        "record_type", "placement_id", "source_state_sha256", "condition",
        "horizon_actions", "action_index", "base_outcome", "detour_outcome",
        "hold_outcome", "preferred_option", "intervention_needed",
        "recovery_open", "loss_control", "fail_safe_contract_violation",
        "task_completion_unsolved", "hard_negative", "recovery_interval_index",
        "window_closure_horizon_actions", "actions_to_window_closure",
        "old_router_advantage", "old_router_detour_advantage",
        "old_router_hold_advantage", "old_router_candidate_option",
        "old_router_pointwise_lcb", "trajectory_id", "decision_id",
    ]
    keys = {key for row in rows for key in row}
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _write_map(
    path: Path,
    trajectories: Sequence[Mapping[str, Any]],
    horizon_grid: Sequence[int],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    rows = sorted(trajectories, key=lambda row: row["placement_id"])
    horizons = list(map(int, horizon_grid))
    option_index = {name: index for index, name in enumerate(DISPLAY_OPTIONS)}
    values = np.full((len(rows), len(horizons)), np.nan)
    advantage = np.full_like(values, np.nan)
    violations = np.zeros_like(values, dtype=bool)
    for y, trajectory in enumerate(rows):
        by_horizon = {int(row["horizon_actions"]): row for row in trajectory["records"]}
        for x, horizon in enumerate(horizons):
            record = by_horizon.get(horizon)
            if record is None:
                continue
            values[y, x] = option_index[record["preferred_option"]]
            advantage[y, x] = float(record["old_router_advantage"])
            violations[y, x] = bool(record["fail_safe_contract_violation"])

    figure, axis = plt.subplots(
        figsize=(max(12, len(horizons) * 0.85), max(4.5, len(rows) * 0.8 + 2.2))
    )
    cmap = ListedColormap(["#4C78A8", "#2A9D6F", "#E9A23B"])
    masked = np.ma.masked_invalid(values)
    axis.imshow(masked, aspect="auto", interpolation="none", cmap=cmap, vmin=-0.5, vmax=2.5)
    for y in range(len(rows)):
        for x in range(len(horizons)):
            if np.isnan(values[y, x]):
                axis.text(x, y, "missing", ha="center", va="center", fontsize=7, color="#666666")
                continue
            text = f"{DISPLAY_OPTIONS[int(values[y, x])][0]}\nΔ={advantage[y, x]:.2f}"
            if violations[y, x]:
                text += "\n!hold"
            axis.text(x, y, text, ha="center", va="center", fontsize=7, color="white")
    axis.set_xticks(range(len(horizons)), [str(value) for value in horizons])
    axis.set_yticks(
        range(len(rows)),
        [row["placement_id"].removeprefix("glass_recovery_") for row in rows],
    )
    axis.set_xlabel("Actions before matched Base collision (H)")
    axis.set_ylabel("Trajectory")
    axis.set_title("P3 dense recovery-window labels; cell text includes old Router advantage")
    axis.set_xticks(np.arange(-0.5, len(horizons), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.legend(
        handles=[Patch(color=cmap(index), label=name) for index, name in enumerate(DISPLAY_OPTIONS)],
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _fmt_fraction(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _write_report(result: Mapping[str, Any], path: Path) -> None:
    trajectories = result["trajectories"]
    missed = {
        row["placement_id"]: row for row in trajectories
        if row["placement_id"] in MISSED_RECOVERABLE_PLACEMENTS
    }
    transitions = [
        row for row in trajectories if row["has_recoverable_to_loss_control_transition"]
    ]
    nonmonotonic = [row for row in trajectories if row["nonmonotonic_recovery"]]
    violations = sum(row["fail_safe_contract_violations"] for row in trajectories)
    overlap = result["feature_overlap"]
    repository = result["provenance"]["capture_repository"]
    lines = [
        "# P3.0 Dense Recovery-Window Supervision",
        "",
        "Status: **development supervision authored; no model training or Router retuning performed**.",
        "",
        f"Quest job: `{result['provenance']['quest_job_id'] or 'not recorded'}`.  "
        f"Collection commit: `{repository['git_commit']}`.",
        "",
        "The five selected trajectories were scanned once under the glass condition. "
        "Every dense anchor branches from the corresponding exact serialized "
        "simulator/controller state into BaseContinue, DetourComplete, and the "
        "zero-delta FailSafeHold. Directional RetreatHold was not used.",
        "",
        "## Dense trajectory map",
        "",
        "| Trajectory | Recovery-open anchors | Intervals (earliest→latest H) | Latest recoverable | Preferred-option transitions | Notes |",
        "|---|---|---|---:|---|---|",
    ]
    for row in trajectories:
        intervals = "; ".join(
            f"{item['earliest_recoverable_anchor']}→{item['latest_recoverable_anchor']} "
            f"({item['recovery_anchor_count']} anchors, span {item['recovery_window_span_actions']})"
            for item in row["recovery_intervals"]
        ) or "none"
        notes = []
        if row["nonmonotonic_recovery"]:
            notes.append("non-monotonic")
        if row["library_unsolved"]:
            notes.append("library-unsolved")
        if row["fail_safe_contract_violations"]:
            notes.append(f"{row['fail_safe_contract_violations']} Hold violation(s)")
        if row["missing_horizons"]:
            notes.append(f"missing H={row['missing_horizons']}")
        lines.append(
            f"| {row['placement_id'].removeprefix('glass_recovery_')} | "
            f"{row['recovery_open_horizons'] or 'none'} | {intervals} | "
            f"{row['latest_recoverable_anchor'] if row['latest_recoverable_anchor'] is not None else 'n/a'} | "
            f"{row['preferred_option_transition_string']} | {', '.join(notes) or '—'} |"
        )

    lines.extend(["", "## Required questions", ""])
    for placement_id in MISSED_RECOVERABLE_PLACEMENTS:
        row = missed.get(placement_id)
        latest = None if row is None else row["latest_recoverable_anchor"]
        lines.append(
            f"- `{placement_id.removeprefix('glass_recovery_')}` latest recoverable "
            f"action: **{latest if latest is not None else 'not observed'} actions before collision**."
        )
    lines.extend([
        f"- Recoverable → loss-control temporal conversion: **{'yes' if transitions else 'no'}**"
        + (" on " + ", ".join(row["placement_id"].removeprefix("glass_recovery_") for row in transitions) if transitions else "")
        + ".",
        f"- Recovery continuity: {sum(row['recovery_windows_contiguous'] for row in trajectories)}/"
        f"{len(trajectories)} trajectories have at most one interval; "
        f"non-monotonic recovery is {'present on ' + ', '.join(row['placement_id'].removeprefix('glass_recovery_') for row in nonmonotonic) if nonmonotonic else 'not observed'}.",
        f"- FailSafeHold contract violations: **{violations}**. All violating anchors, if any, are retained.",
        "",
        "## Hard controls and feature overlap",
        "",
        f"The authoring selected {result['coverage']['hard_control_regions']} control regions "
        f"({result['coverage']['hard_negative_records']} Base-success states) from the union "
        "of the highest-peak and longest pointwise-margin exceedances in the existing P2.5 Base traces. "
        "No new control option rollout was run.",
        "",
    ])
    if overlap["available"]:
        lines.extend([
            f"Overlap is measured in the {overlap['dimensions']}-D standardized frozen-Router "
            "prediction-output space. Leave-one-out 1-NN balanced accuracy is "
            f"`{overlap['leave_one_out_1nn_balanced_accuracy']:.3f}`; "
            f"{_fmt_fraction(overlap['recovery_points_with_hard_neighbor_no_farther_than_recovery_neighbor_fraction'])} "
            "of recovery-open points have a hard-control neighbor no farther than their nearest "
            "other recovery-open point. Median recovery→hard distance is "
            f"`{overlap['median_recovery_to_hard_distance']:.3f}` versus median "
            f"recovery→recovery distance `{_fmt_fraction(overlap['median_recovery_to_recovery_distance'])}`. "
            "These diagnostics quantify overlap; they are not a newly tuned decision rule.",
            "",
        ])
    else:
        lines.extend([f"Overlap unavailable: {overlap['reason']}.", ""])
    lines.extend([
        "## Direct-supervision targets",
        "",
        "The next model should learn three related but distinct targets from causal history: "
        "(1) whether intervention is needed relative to Base, (2) whether Detour task completion "
        "is still available and the actions remaining to the local window closure, and (3) the "
        "preferred option among Base, Detour, and FailSafeHold. Loss-control and Hold-contract "
        "violation should remain explicit auxiliary labels rather than being folded into one "
        "binary hazard score. Hard Base-success regions must be included as intervention-negative "
        "examples because the old Router assigns them sustained high advantage.",
        "",
        "## Outputs and provenance",
        "",
        f"- Dense exact-state anchors: {result['coverage']['dense_anchor_records']} "
        f"({result['coverage']['complete_dense_trajectories']}/{result['coverage']['dense_trajectories']} complete grids).",
        f"- Recovery-open anchors: {result['coverage']['recovery_open_records']}; "
        f"loss-control anchors: {result['coverage']['loss_control_records']}; "
        f"Base-preferred anchors: {result['coverage']['base_preferred_records']}.",
        f"- Capture root: `{result['provenance']['capture_root']}`.",
        f"- Hard-control source: `{result['provenance']['hard_control_root']}`.",
        "- Collection command: `python scripts/collect_counterfactual_option_rollouts.py "
        "--conditions glass --history-length 8 --horizons 30 28 ... 4 2 1 "
        "--retreat-mode hold ...`",
        "- Authoring command: `python scripts/author_recovery_window_supervision.py "
        "--capture <P3 capture> --router-model <frozen Router> "
        "--hard-control-capture <P2 exact-prefix capture> ...`",
        "",
        "Completion here is data authoring only. No temporal model, new Router, dynamic "
        "evaluation, alpha, margin, or intervention threshold was trained or adjusted.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def author(args: argparse.Namespace) -> dict[str, Any]:
    capture = args.capture.resolve()
    hard_root = args.hard_control_capture.resolve()
    manifest_path = capture / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest["protocol"]
    if protocol.get("conditions") != ["glass"]:
        raise ValueError("P3 dense capture must contain only the glass condition")
    if protocol.get("retreat", {}).get("mode") != "hold":
        raise ValueError("P3 requires zero-delta FailSafeHold, not directional RetreatHold")
    horizons = list(map(int, args.horizons))
    if list(map(int, protocol.get("horizons", []))) != horizons:
        raise ValueError("capture horizon grid does not match the requested dense grid")
    if int(protocol.get("history_length", -1)) != 8:
        raise ValueError("P3 capture must preserve history-length=8")

    data = load_capture(capture, catastrophe_cost=args.catastrophe_cost)
    if set(data["conditions"].tolist()) != {"glass"}:
        raise ValueError("non-glass decision rows are not allowed in dense supervision")
    router = FrozenOutcomeRouter.load(args.router_model)
    point = _router_point(router, args.catastrophe_cost, args.target_intervention_rate)
    dense, dense_vectors = _score_dense_rows(
        data, router,
        catastrophe_cost=args.catastrophe_cost,
        margin=float(point["delta"]),
    )
    trajectories = _trajectory_summaries(dense, horizons)

    hard_manifest_path = hard_root / "capture_manifest.json"
    hard_manifest = json.loads(hard_manifest_path.read_text())
    hard_margin = float(hard_manifest["router_point"]["pointwise_margin"])
    if not np.isclose(hard_margin, float(point["delta"])):
        raise ValueError("P2.5 hard-control trace and frozen Router point disagree")
    hard, hard_regions, hard_vectors = extract_hard_control_records(
        _read_jsonl(hard_root / "dynamic_episodes.jsonl"),
        _read_jsonl(hard_root / "router_trace.jsonl"),
        pointwise_margin=hard_margin,
        top_by_peak=args.hard_regions_by_peak,
        top_by_duration=args.hard_regions_by_duration,
    )
    all_records = sorted(
        dense, key=lambda row: (row["placement_id"], -int(row["horizon_actions"]))
    ) + hard
    recovery_vectors = [
        dense_vectors[row["decision_id"]] for row in dense if row["recovery_open"]
    ]
    selected_hard_vectors = [hard_vectors[row["decision_id"]] for row in hard]
    overlap = feature_overlap_summary(recovery_vectors, selected_hard_vectors)

    result = {
        "schema_version": 1,
        "kind": "p3_dense_recovery_window_supervision",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_supervision_no_model_training",
        "protocol": {
            "horizons": horizons,
            "condition": "glass",
            "history_length": 8,
            "options": list(DISPLAY_OPTIONS),
            "raw_compatibility_slots": list(OPTIONS),
            "third_option_implementation": "FailSafeHold_zero_delta",
            "outcome_priority": ["task_success", "safe_noncompletion", "catastrophe"],
            "tie_breaks": [
                "Base wins when Base and Detour both succeed",
                "FailSafeHold wins when no option succeeds and Hold is safe_noncompletion",
                "remaining equal-outcome ties prefer Base, then Detour, then FailSafeHold",
            ],
            "recovery_interval_definition": "adjacent positive anchors on the predeclared sampled horizon grid",
            "window_span_definition": "earliest recoverable H minus latest recoverable H",
            "hard_negative_definition": "Base-success P2.5 states in top peak or top duration old-Router regions above the pointwise margin",
        },
        "old_router_point": point,
        "coverage": {
            "dense_trajectories": len(trajectories),
            "complete_dense_trajectories": sum(row["complete_dense_grid"] for row in trajectories),
            "dense_anchor_records": len(dense),
            "recovery_open_records": sum(row["recovery_open"] for row in dense),
            "loss_control_records": sum(row["loss_control"] for row in dense),
            "base_preferred_records": sum(row["preferred_option"] == "Base" for row in dense),
            "fail_safe_contract_violations": sum(row["fail_safe_contract_violation"] for row in dense),
            "hard_control_regions": len(hard_regions),
            "hard_negative_records": len(hard),
            "all_records": len(all_records),
        },
        "hard_control_regions": hard_regions,
        "feature_overlap": overlap,
        "provenance": {
            "quest_job_id": args.quest_job_id,
            "capture_root": str(capture),
            "capture_repository": manifest["repository"],
            "capture_artifacts": {
                name: _artifact(capture / name) for name in (
                    "capture_manifest.json", "decision_metadata.json",
                    "option_rollouts.jsonl", "decision_features.npz",
                )
            },
            "router_model": _artifact(args.router_model.resolve()),
            "router_artifact": _artifact(
                args.router_model.resolve().parent / router.manifest["artifact_npz"]
            ),
            "hard_control_root": str(hard_root),
            "hard_control_repository": hard_manifest["repository"],
            "hard_control_artifacts": {
                name: _artifact(hard_root / name) for name in (
                    "capture_manifest.json", "dynamic_episodes.jsonl", "router_trace.jsonl"
                )
            },
        },
        "trajectories": trajectories,
    }

    outputs = [
        args.output_jsonl, args.output_trajectories, args.output_csv,
        args.output_map, args.output_report,
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {path}; pass --overwrite")
    _write_jsonl(args.output_jsonl, all_records)
    args.output_trajectories.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_csv, all_records)
    _write_map(args.output_map, trajectories, horizons)
    _write_report(result, args.output_report)
    print(json.dumps({
        "coverage": result["coverage"],
        "outputs": [str(path) for path in outputs],
    }, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--router-model", type=Path, required=True)
    parser.add_argument("--hard-control-capture", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DENSE_HORIZONS))
    parser.add_argument("--catastrophe-cost", type=float, default=1.0)
    parser.add_argument("--target-intervention-rate", type=float, default=0.4)
    parser.add_argument("--hard-regions-by-peak", type=int, default=2)
    parser.add_argument("--hard-regions-by-duration", type=int, default=2)
    parser.add_argument("--quest-job-id", default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument(
        "--output-report", type=Path,
        default=Path("results/P3_RECOVERY_WINDOW_SUPERVISION_DEV_20260819.md"),
    )
    parser.add_argument(
        "--output-jsonl", type=Path,
        default=Path("results/p3_recovery_window_records_20260819.jsonl"),
    )
    parser.add_argument(
        "--output-trajectories", type=Path,
        default=Path("results/p3_recovery_window_trajectories_20260819.json"),
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=Path("results/p3_recovery_window_records_20260819.csv"),
    )
    parser.add_argument(
        "--output-map", type=Path,
        default=Path("results/p3_recovery_window_map_20260819.png"),
    )
    parser.add_argument("--overwrite", action="store_true")
    author(parser.parse_args())


if __name__ == "__main__":
    main()
