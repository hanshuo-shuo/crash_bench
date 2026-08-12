#!/usr/bin/env python3
"""Fit and calibrate the source-disjoint D0 glass detector.

The input capture must contain frozen hidden states, 8-D robot state, 7-D
nominal actions, and per-frame metadata produced by the revised glass capture
job.  A separately frozen source-split manifest prevents frame/episode leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.glass_detector import (
    CONTROL_CONDITIONS,
    DETECTOR_HORIZON_ACTIONS,
    DETECTOR_SPLITS,
    DETECTOR_VARIANTS,
    auc,
    episode_operating_metrics,
    fit_glass_detector,
    risk_frame_targets,
    split_indices,
    validate_capture_arrays,
    validate_source_split,
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_capture(path: str | Path, meta_path: str | Path):
    with np.load(path, allow_pickle=False) as archive:
        hidden_key = "hidden" if "hidden" in archive.files else "H"
        missing = {"robot_state", "nominal_action"} - set(archive.files)
        if missing:
            raise ValueError(
                "capture predates deployable D0 features; recapture robot state and "
                f"nominal action (missing {sorted(missing)})"
            )
        hidden = np.asarray(archive[hidden_key], dtype=np.float32)
        robot_state = np.asarray(archive["robot_state"], dtype=np.float32)
        nominal_action = np.asarray(archive["nominal_action"], dtype=np.float32)
    rows = json.loads(Path(meta_path).read_text())
    if not isinstance(rows, list):
        raise ValueError("glass detector metadata must be a JSON list of frame rows")
    validate_capture_arrays(hidden, robot_state, nominal_action, rows)
    return hidden, robot_state, nominal_action, rows


def _load_source_splits(path: str | Path, horizon_actions: int) -> Mapping[str, str]:
    payload = json.loads(Path(path).read_text())
    if payload.get("kind") != "glass_detector_source_split":
        raise ValueError("split manifest kind must be glass_detector_source_split")
    if int(payload.get("horizon_actions", -1)) != int(horizon_actions):
        raise ValueError("split manifest horizon disagrees with detector horizon")
    source_splits = payload.get("source_splits")
    if not isinstance(source_splits, dict):
        raise ValueError("split manifest requires a source_splits mapping")
    return {str(source): str(split) for source, split in source_splits.items()}


def _json_number(value: float) -> float | None:
    return None if not math.isfinite(float(value)) else float(value)


def fit(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    if int(args.horizon_actions) != DETECTOR_HORIZON_ACTIONS:
        raise ValueError(
            f"D0 is frozen at H={DETECTOR_HORIZON_ACTIONS}; got {args.horizon_actions}"
        )

    hidden, robot_state, nominal_action, rows = _load_capture(
        args.capture, args.metadata
    )
    source_splits = _load_source_splits(args.split_manifest, args.horizon_actions)
    sources_by_split = validate_source_split(rows, source_splits)
    targets, usable = risk_frame_targets(rows, horizon_actions=args.horizon_actions)
    train_rows = split_indices(rows, source_splits, "train") & usable
    if not train_rows.any():
        raise ValueError("source-disjoint train split has no usable frames")

    input_provenance = {
        "capture": str(args.capture),
        "capture_sha256": _file_sha256(args.capture),
        "metadata": str(args.metadata),
        "metadata_sha256": _file_sha256(args.metadata),
        "source_split_manifest": str(args.split_manifest),
        "source_split_manifest_sha256": _file_sha256(args.split_manifest),
    }
    variants = {}
    calibration_mask = split_indices(rows, source_splits, "calibration")
    development_mask = split_indices(rows, source_splits, "development")
    artifact_names = {
        "hidden_robot_action": "glass_detector.npz",
        "hidden_only": "baseline_hidden_only.npz",
        "robot_action_only": "baseline_robot_action_only.npz",
    }
    for variant in DETECTOR_VARIANTS:
        detector = fit_glass_detector(
            hidden[train_rows],
            robot_state[train_rows],
            nominal_action[train_rows],
            targets[train_rows],
            variant=variant,
            pca_components=args.pca_components,
            l2=args.l2,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            seed=args.seed,
            metadata={
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "horizon_actions": int(args.horizon_actions),
                "feature_contract": variant,
                "source_disjoint": True,
                "threshold_selected_split": "calibration",
                "development_used_for_threshold": False,
                "source_state_counts": {
                    split: len(sources) for split, sources in sources_by_split.items()
                },
                **input_provenance,
            },
        )
        all_scores = np.asarray(
            detector.score(hidden, robot_state, nominal_action), dtype=float
        )
        calibration = episode_operating_metrics(
            all_scores,
            rows,
            calibration_mask,
            horizon_actions=args.horizon_actions,
            max_control_episode_fpr=args.max_control_episode_fpr,
        )
        detector = detector.with_threshold(calibration["threshold"], calibration)
        development = episode_operating_metrics(
            all_scores,
            rows,
            development_mask,
            threshold=detector.threshold,
            horizon_actions=args.horizon_actions,
            max_control_episode_fpr=args.max_control_episode_fpr,
        )
        frame_auc = {}
        for split in DETECTOR_SPLITS:
            mask = split_indices(rows, source_splits, split) & usable
            frame_auc[split] = _json_number(auc(all_scores[mask], targets[mask]))
        artifact = output / artifact_names[variant]
        detector.save(artifact)
        variants[variant] = {
            "artifact": str(artifact),
            "artifact_sha256": _file_sha256(artifact),
            "frame_auc_by_split": frame_auc,
            "calibration": calibration,
            "development_at_frozen_threshold": development,
        }
        print(json.dumps({"variant": variant, **variants[variant]}, sort_keys=True))

    primary = variants["hidden_robot_action"]
    calibration = primary["calibration"]
    controls_present = all(
        calibration["control_episode_fpr_by_condition"][condition] is not None
        for condition in CONTROL_CONDITIONS
    )
    decision_checks = {
        "timely_trigger_rate_at_least_minimum": (
            calibration["timely_trigger_rate"] >= args.min_timely_trigger_rate
        ),
        "control_episode_fpr_at_most_maximum": (
            calibration["control_episode_fpr"] <= args.max_control_episode_fpr
        ),
        "control_fpr_resolution_supports_bound": (
            calibration["control_fpr_resolution"] <= args.max_control_episode_fpr
        ),
        "both_offpath_and_noglass_controls_present": controls_present,
        "minimum_calibration_source_states": (
            len(sources_by_split["calibration"])
            >= args.min_calibration_source_states
        ),
        "positive_operating_point_exists": calibration["has_positive_operating_point"],
    }
    summary = {
        "schema_version": 1,
        "kind": "glass_detector_d0_result",
        "status": (
            "ready_for_development_online_d1"
            if all(decision_checks.values())
            else "not_ready_for_development_online_d1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_actions": int(args.horizon_actions),
        "primary_variant": "hidden_robot_action",
        "baselines": ["hidden_only", "robot_action_only"],
        "source_disjoint": True,
        "sources_by_split": sources_by_split,
        "input_provenance": input_provenance,
        "operating_gate": {
            "minimum_timely_trigger_rate": args.min_timely_trigger_rate,
            "maximum_control_episode_fpr": args.max_control_episode_fpr,
            "minimum_calibration_source_states": args.min_calibration_source_states,
            "checks": decision_checks,
            "pass": all(decision_checks.values()),
        },
        "variants": variants,
    }
    summary_path = output / "d0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon-actions", type=int, default=DETECTOR_HORIZON_ACTIONS)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--l2", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-control-episode-fpr", type=float, default=0.10)
    parser.add_argument("--min-timely-trigger-rate", type=float, default=0.80)
    parser.add_argument("--min-calibration-source-states", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.pca_components < 1 or args.iterations < 1:
        raise SystemExit("PCA components and iterations must be positive")
    if not 0.0 <= args.max_control_episode_fpr <= 1.0:
        raise SystemExit("maximum control FPR must lie in [0, 1]")
    if not 0.0 <= args.min_timely_trigger_rate <= 1.0:
        raise SystemExit("minimum timely trigger rate must lie in [0, 1]")
    if args.min_calibration_source_states < 1:
        raise SystemExit("minimum calibration source states must be positive")
    fit(args)


if __name__ == "__main__":
    main()
