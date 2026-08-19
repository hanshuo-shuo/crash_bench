#!/usr/bin/env python3
"""Freeze the one P3.2 source-level boundary for the direct routing head.

The calibration cohort and statistical rule are inherited from P2.  The only
score change is the already-frozen direct head margin
``max(logit_Detour, logit_Hold) - logit_Base``.  Alpha is fixed at 0.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.counterfactual_router import (
    FrozenOutcomeRouter,
    classify_option_outcome,
    split_conformal_upper_boundary,
)
from crashbench.direct_recovery_router import (
    DIRECT_INPUT_DIMENSION,
    DIRECT_OPTIONS,
    DirectRecoveryWindowHead,
    router_output_feature,
)
from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import array_sha256, read_placement_manifest
from crashbench.provenance import repository_provenance, require_checkpoint_revision
from scripts.capture_glass_detector_placements import _seed_everything
from scripts.collect_counterfactual_option_rollouts import _scan_condition
from scripts.collect_glass_recovery_pairs import CandidateRejected


SEQUENTIAL_ALPHA = 0.1
CALIBRATION_SPLITS = ("validation", "heldout")
CONTROL_CONDITIONS = ("offpath", "noglass")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _head_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "schema_version": int(archive["schema_version"]),
            "kind": str(archive["kind"]),
            "candidate_status": str(archive["status"]),
            "classes": [str(value) for value in archive["classes"].tolist()],
            "input_feature_field": str(archive["input_feature_field"]),
            "input_dimension": int(archive["input_dimension"]),
            "l2": float(archive["l2"]),
            "training_records": int(archive["training_records"]),
            "training_sources": int(archive["training_sources"]),
            "training_records_sha256": str(archive["training_records_sha256"]),
            "optimizer_converged": bool(archive["optimizer_converged"]),
            "feature_mean": archive["feature_mean"].tolist(),
            "feature_scale": archive["feature_scale"].tolist(),
            "head_weight": archive["head_weight"].tolist(),
            "head_bias": archive["head_bias"].tolist(),
            "array_sha256": {
                name: _array_sha256(archive[name])
                for name in (
                    "feature_mean", "feature_scale", "head_weight", "head_bias"
                )
            },
        }


def _trajectory_scores(
    feature_router: FrozenOutcomeRouter,
    direct_head: DirectRecoveryWindowHead,
    scan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], float, int, int]:
    trace = []
    for action_index, frame in enumerate(scan["rows"]):
        prediction = feature_router.predict(
            frame["hidden"], frame["robot_state"], frame["nominal_action"]
        )
        feature = router_output_feature(prediction)
        logits = np.asarray(direct_head.logits(feature), dtype=np.float64)
        probabilities = np.asarray(
            direct_head.probabilities(feature), dtype=np.float64
        )
        candidate = 1 + int(np.argmax(logits[1:]))
        margin = float(logits[candidate] - logits[0])
        trace.append({
            "action_index": action_index,
            "router_output_feature": feature.tolist(),
            "direct_logits": logits.tolist(),
            "direct_probabilities": probabilities.tolist(),
            "candidate_option": DIRECT_OPTIONS[candidate],
            "direct_nonbase_logit_margin": margin,
        })
    scores = np.asarray(
        [row["direct_nonbase_logit_margin"] for row in trace], dtype=np.float64
    )
    maximum_index = int(np.argmax(scores))
    maximum_logits = np.asarray(trace[maximum_index]["direct_logits"])
    candidate = 1 + int(np.argmax(maximum_logits[1:]))
    return trace, float(scores[maximum_index]), maximum_index, candidate


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    output.mkdir(parents=True)

    repo_root = Path(__file__).resolve().parents[1]
    repository = repository_provenance(repo_root, require_clean=True)
    declared = os.environ.get("CB_CODE_COMMIT")
    if declared is not None and declared != repository["git_commit"]:
        raise SystemExit("CB_CODE_COMMIT does not match checked-out source")
    revision = require_checkpoint_revision(args.checkpoint_revision)

    placements_path = args.placements.resolve()
    placements, placement_payload = read_placement_manifest(placements_path)
    placements = [
        placement for placement in placements
        if placement.split in set(CALIBRATION_SPLITS)
    ]
    old_capture_path = args.feature_router_capture_manifest.resolve()
    old_capture = json.loads(old_capture_path.read_text(encoding="utf-8"))
    expected_sources = {
        str(source)
        for source, split in old_capture["source_splits"].items()
        if split == "calibration"
    }
    selected_sources = {placement.source_state_sha256 for placement in placements}
    if selected_sources != expected_sources:
        raise RuntimeError(
            "P3.2 calibration sources differ from the original Router calibration split"
        )

    head_path = args.direct_head_model.resolve()
    supervision_path = args.supervision_records.resolve()
    head_metadata = _head_metadata(head_path)
    if head_metadata["classes"] != list(DIRECT_OPTIONS):
        raise ValueError("direct head option mapping is not frozen")
    if head_metadata["input_dimension"] != DIRECT_INPUT_DIMENSION:
        raise ValueError("direct head input is not frozen at 10 dimensions")
    if head_metadata["training_records_sha256"] != _sha256(supervision_path):
        raise ValueError("direct head does not match the P3.0 supervision artifact")
    supervision_sources = {
        str(row["source_state_sha256"]) for row in _read_jsonl(supervision_path)
    }
    overlap = supervision_sources & expected_sources
    if overlap:
        raise RuntimeError(
            f"direct supervision overlaps sequential calibration sources: {sorted(overlap)}"
        )

    feature_router_path = args.feature_router_model.resolve()
    feature_router = FrozenOutcomeRouter.load(feature_router_path)
    direct_head = DirectRecoveryWindowHead.load(head_path)
    feature_router_artifact = (
        feature_router_path.parent / str(feature_router.manifest["artifact_npz"])
    )

    freeze = {
        "schema_version": 1,
        "kind": "p3_2_frozen_direct_recovery_window_head",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen before the single P3.2 dynamic closeout",
        "repository": repository,
        "model": {
            "path": str(head_path),
            "sha256": _sha256(head_path),
            **head_metadata,
        },
        "supervision": {
            "path": str(supervision_path),
            "sha256": _sha256(supervision_path),
            "records": head_metadata["training_records"],
            "sources": head_metadata["training_sources"],
        },
        "input_contract": {
            "dimension": DIRECT_INPUT_DIMENSION,
            "field": "router_output_feature",
            "order": (
                "row-major 3 options x [task_success, catastrophe, safe_noncompletion] "
                "probabilities, then Base catastrophe probability"
            ),
            "feature_router_model": str(feature_router_path),
            "feature_router_model_sha256": _sha256(feature_router_path),
            "feature_router_artifact": str(feature_router_artifact),
            "feature_router_artifact_sha256": _sha256(feature_router_artifact),
        },
        "option_mapping": {
            "0": "Base",
            "1": "Detour",
            "2": "FailSafeHold",
        },
        "decision_before_sequential_calibration": (
            "linear logits only; no class margin or probability threshold"
        ),
        "tunable_head_parameters_remaining": [],
    }
    freeze_path = output / "direct_head_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from crashbench.policies import OpenVLAPolicy

    policy = OpenVLAPolicy(
        pretrained_checkpoint=args.checkpoint,
        checkpoint_revision=revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        capture_hidden=True,
    )
    envs: dict[tuple[str, int], LiberoEnv] = {}
    trajectory_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for placement in placements:
        source_path = placements_path.parent / placement.source_state_path
        source_state = np.load(source_path, allow_pickle=False)
        if array_sha256(source_state) != placement.source_state_sha256:
            raise ValueError(f"{placement.placement_id}: source-state hash mismatch")
        env_key = (placement.task_suite, int(placement.task_id))
        if env_key not in envs:
            envs[env_key] = LiberoEnv(*env_key, seed=args.rollout_seed)
        env = envs[env_key]
        for condition in CONTROL_CONDITIONS:
            _seed_everything(args.rollout_seed)
            env.seed(args.rollout_seed)
            policy.reset()
            try:
                scan = _scan_condition(
                    env,
                    policy,
                    source_state,
                    placement,
                    condition,
                    settle_steps=args.settle_steps,
                    max_steps=args.scan_steps,
                )
            except CandidateRejected as exc:
                exclusions.append({
                    "placement_id": placement.placement_id,
                    "source_state_sha256": placement.source_state_sha256,
                    "condition": condition,
                    "reason": exc.reason,
                })
                continue
            trace, maximum, maximum_index, candidate = _trajectory_scores(
                feature_router, direct_head, scan
            )
            trajectory_id = f"{placement.source_state_sha256}:{condition}"
            for trace_row in trace:
                _append_jsonl(output / "calibration_direct_trace.jsonl", {
                    "trajectory_id": trajectory_id,
                    "placement_id": placement.placement_id,
                    "source_state_sha256": placement.source_state_sha256,
                    "condition": condition,
                    **trace_row,
                })
            base_outcome = classify_option_outcome(
                crashed=bool(scan["crashed"]), succeeded=bool(scan["succeeded"])
            )
            row = {
                "trajectory_id": trajectory_id,
                "placement_id": placement.placement_id,
                "source_state_sha256": placement.source_state_sha256,
                "condition": condition,
                "base_outcome": base_outcome,
                "eligible_base_success_control": base_outcome == "task_success",
                "actions": len(trace),
                "max_direct_nonbase_logit_margin": maximum,
                "max_action_index": maximum_index,
                "max_option": DIRECT_OPTIONS[candidate],
            }
            trajectory_rows.append(row)
            _append_jsonl(output / "calibration_trajectories.jsonl", row)
        print(
            f"source={placement.source_state_sha256[:10]} "
            f"placement={placement.placement_id} direct calibration complete",
            flush=True,
        )

    eligible = [row for row in trajectory_rows if row["eligible_base_success_control"]]
    source_maxima = {
        source: max(
            float(row["max_direct_nonbase_logit_margin"])
            for row in eligible if row["source_state_sha256"] == source
        )
        for source in sorted({row["source_state_sha256"] for row in eligible})
    }
    if set(source_maxima) != expected_sources:
        missing = sorted(expected_sources - set(source_maxima))
        raise RuntimeError(f"calibration sources without a Base-success control: {missing}")
    sequential_boundary = split_conformal_upper_boundary(
        list(source_maxima.values()), alpha=SEQUENTIAL_ALPHA
    )
    rank = int(np.ceil((len(source_maxima) + 1) * (1.0 - SEQUENTIAL_ALPHA)))
    boundary = {
        "schema_version": 1,
        "kind": "p3_2_direct_head_source_level_sequential_boundary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen before the single P3.2 dynamic closeout",
        "repository": repository,
        "direct_head_freeze": str(freeze_path),
        "direct_head_freeze_sha256": _sha256(freeze_path),
        "direct_head_model_sha256": _sha256(head_path),
        "feature_router_model_sha256": _sha256(feature_router_path),
        "score": "max(logit_Detour, logit_FailSafeHold) - logit_Base",
        "alpha": SEQUENTIAL_ALPHA,
        "strict_crossing": True,
        "source_conformal_rank": rank,
        "calibration_sources": len(source_maxima),
        "eligible_control_trajectories": len(eligible),
        "source_maxima": source_maxima,
        "sequential_boundary": float(sequential_boundary),
        "calibration_source_false_crossings": int(sum(
            value > sequential_boundary for value in source_maxima.values()
        )),
        "runtime_rule": (
            "Base until the first strict margin crossing; at crossing choose the "
            "larger of Detour/Hold logits and latch"
        ),
        "alpha_sweep_performed": False,
        "tunable_parameters_remaining": [],
    }
    boundary_path = output / "sequential_boundary.json"
    boundary_path.write_text(
        json.dumps(boundary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "kind": "p3_2_direct_head_sequential_calibration_capture",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": revision,
        "checkpoint_identity": policy.checkpoint_identity,
        "placements": str(placements_path),
        "placements_sha256": _sha256(placements_path),
        "placement_design_metadata": placement_payload.get("metadata", {}),
        "splits": list(CALIBRATION_SPLITS),
        "conditions": list(CONTROL_CONDITIONS),
        "source_states": len(selected_sources),
        "trajectories": len(trajectory_rows),
        "eligible_base_success_controls": len(eligible),
        "exclusions": exclusions,
        "artifacts": {
            "freeze": "direct_head_freeze.json",
            "boundary": "sequential_boundary.json",
            "trajectories": "calibration_trajectories.jsonl",
            "trace": "calibration_direct_trace.jsonl",
        },
    }
    (output / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "sources": len(source_maxima),
        "eligible_controls": len(eligible),
        "alpha": SEQUENTIAL_ALPHA,
        "sequential_boundary": sequential_boundary,
    }, indent=2, sort_keys=True))
    return boundary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placements", required=True, type=Path)
    parser.add_argument("--feature-router-model", required=True, type=Path)
    parser.add_argument(
        "--feature-router-capture-manifest", required=True, type=Path
    )
    parser.add_argument("--direct-head-model", required=True, type=Path)
    parser.add_argument("--supervision-records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--scan-steps", type=int, default=220)
    args = parser.parse_args()
    calibrate(args)


if __name__ == "__main__":
    main()
