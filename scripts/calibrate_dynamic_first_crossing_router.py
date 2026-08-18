#!/usr/bin/env python3
"""Freeze a source-level sequential boundary for dynamic first crossing.

The calibration unit is one source state.  For each source we take the maximum
predicted non-Base advantage over every action in every Base-success control
trajectory, then apply a one-sided split-conformal rank across sources.
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
    OPTIONS,
    classify_option_outcome,
    option_utilities,
    split_conformal_upper_boundary,
)
from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import array_sha256, read_placement_manifest
from crashbench.provenance import repository_provenance, require_checkpoint_revision
from scripts.capture_glass_detector_placements import _seed_everything
from scripts.collect_counterfactual_option_rollouts import _scan_condition
from scripts.collect_glass_recovery_pairs import CandidateRejected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _trajectory_scores(
    router: FrozenOutcomeRouter,
    scan: Mapping[str, Any],
    *,
    catastrophe_cost: float,
) -> tuple[list[dict[str, Any]], float, int, int]:
    trace = []
    for action_index, feature in enumerate(scan["rows"]):
        prediction = router.predict(
            feature["hidden"], feature["robot_state"], feature["nominal_action"]
        )
        probability = np.asarray(
            prediction["option_outcome_probabilities"], dtype=np.float64
        )
        utility = option_utilities(probability, catastrophe_cost)
        advantage = utility - utility[0]
        candidate = 1 + int(np.argmax(advantage[1:]))
        trace.append({
            "action_index": action_index,
            "advantage_vs_base": advantage.tolist(),
            "candidate_option": OPTIONS[candidate],
            "max_nonbase_advantage": float(advantage[candidate]),
        })
    scores = np.asarray([row["max_nonbase_advantage"] for row in trace])
    maximum_index = int(np.argmax(scores))
    candidate = 1 + int(np.argmax(
        np.asarray(trace[maximum_index]["advantage_vs_base"])[1:]
    ))
    return trace, float(scores[maximum_index]), maximum_index, candidate


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    repo_root = Path(__file__).resolve().parents[1]
    repo = repository_provenance(repo_root, require_clean=True)
    declared = os.environ.get("CB_CODE_COMMIT")
    if declared is not None and declared != repo["git_commit"]:
        raise SystemExit("CB_CODE_COMMIT does not match checked-out source")
    revision = require_checkpoint_revision(args.checkpoint_revision)

    placements_path = Path(args.placements).resolve()
    placements, placement_payload = read_placement_manifest(placements_path)
    placements = [
        placement for placement in placements if placement.split in set(args.splits)
    ]
    router_path = Path(args.router_model).resolve()
    router = FrozenOutcomeRouter.load(router_path)
    capture_manifest_path = Path(args.router_capture_manifest).resolve()
    capture_manifest = json.loads(capture_manifest_path.read_text())
    expected_sources = {
        source for source, split in capture_manifest["source_splits"].items()
        if split == "calibration"
    }
    selected_sources = {placement.source_state_sha256 for placement in placements}
    if selected_sources != expected_sources:
        raise RuntimeError(
            "selected sequential calibration sources do not match the frozen "
            "router calibration split"
        )

    point = router.manifest["calibration"]["router_frontier"][
        f"lambda_{args.catastrophe_cost:g}"
    ][f"target_{args.target_intervention_rate:.1f}"]
    pointwise_margin = float(point["delta"])

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
        for condition in args.conditions:
            _seed_everything(args.rollout_seed)
            env.seed(args.rollout_seed)
            policy.reset()
            try:
                scan = _scan_condition(
                    env, policy, source_state, placement, condition,
                    settle_steps=args.settle_steps, max_steps=args.scan_steps,
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
                router, scan, catastrophe_cost=args.catastrophe_cost
            )
            trajectory_id = f"{placement.source_state_sha256}:{condition}"
            for trace_row in trace:
                _append_jsonl(output / "calibration_router_trace.jsonl", {
                    "trajectory_id": trajectory_id,
                    "placement_id": placement.placement_id,
                    "source_state_sha256": placement.source_state_sha256,
                    "condition": condition,
                    **trace_row,
                })
            outcome = classify_option_outcome(
                crashed=bool(scan["crashed"]), succeeded=bool(scan["succeeded"])
            )
            row = {
                "trajectory_id": trajectory_id,
                "placement_id": placement.placement_id,
                "source_state_sha256": placement.source_state_sha256,
                "condition": condition,
                "base_outcome": outcome,
                "eligible_base_success_control": outcome == "task_success",
                "actions": len(trace),
                "max_nonbase_advantage": maximum,
                "max_action_index": maximum_index,
                "max_option": OPTIONS[candidate],
            }
            trajectory_rows.append(row)
            _append_jsonl(output / "calibration_trajectories.jsonl", row)
        print(
            f"source={placement.source_state_sha256[:10]} "
            f"placement={placement.placement_id} calibration traces complete",
            flush=True,
        )

    eligible = [row for row in trajectory_rows if row["eligible_base_success_control"]]
    source_maxima = {
        source: max(
            float(row["max_nonbase_advantage"])
            for row in eligible if row["source_state_sha256"] == source
        )
        for source in sorted({row["source_state_sha256"] for row in eligible})
    }
    boundaries = {}
    for alpha in args.alphas:
        sequential_margin = split_conformal_upper_boundary(
            list(source_maxima.values()), alpha=alpha
        )
        effective_margin = max(pointwise_margin, sequential_margin)
        boundaries[f"alpha_{alpha:.1f}"] = {
            "alpha": float(alpha),
            "source_conformal_rank": int(np.ceil(
                (len(source_maxima) + 1) * (1.0 - float(alpha))
            )),
            "sequential_margin": sequential_margin,
            "pointwise_margin": pointwise_margin,
            "effective_margin": effective_margin,
            "calibration_source_false_crossings": int(sum(
                value > effective_margin for value in source_maxima.values()
            )),
        }

    boundary = {
        "schema_version": 1,
        "kind": "source_level_sequential_first_crossing_boundary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repo,
        "router_model": str(router_path),
        "router_model_sha256": _sha256(router_path),
        "router_artifact_sha256": router.manifest["artifact_npz_sha256"],
        "router_capture_manifest": str(capture_manifest_path),
        "router_capture_manifest_sha256": _sha256(capture_manifest_path),
        "catastrophe_cost": float(args.catastrophe_cost),
        "target_intervention_rate": float(args.target_intervention_rate),
        "primary_alpha": float(args.primary_alpha),
        "calibration_unit": (
            "source maximum over all actions, non-Base options, and Base-success "
            "offpath/noglass control trajectories"
        ),
        "strict_crossing": True,
        "calibration_sources": len(source_maxima),
        "eligible_control_trajectories": len(eligible),
        "source_maxima": source_maxima,
        "boundaries": boundaries,
    }
    (output / "sequential_boundary.json").write_text(
        json.dumps(boundary, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "kind": "dynamic_first_crossing_calibration_capture",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repo,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": revision,
        "checkpoint_identity": policy.checkpoint_identity,
        "placements": str(placements_path),
        "placements_sha256": _sha256(placements_path),
        "placement_design_metadata": placement_payload.get("metadata", {}),
        "splits": list(args.splits),
        "conditions": list(args.conditions),
        "source_states": len(selected_sources),
        "trajectories": len(trajectory_rows),
        "eligible_base_success_controls": len(eligible),
        "exclusions": exclusions,
        "artifacts": {
            "trajectories": "calibration_trajectories.jsonl",
            "trace": "calibration_router_trace.jsonl",
            "boundary": "sequential_boundary.json",
        },
    }
    (output / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "output": str(output),
        "sources": len(source_maxima),
        "eligible_controls": len(eligible),
        "boundaries": boundaries,
    }, indent=2, sort_keys=True))
    return boundary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", required=True)
    parser.add_argument("--router-model", required=True)
    parser.add_argument("--router-capture-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["validation", "heldout"])
    parser.add_argument("--conditions", nargs="+", default=["offpath", "noglass"])
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--scan-steps", type=int, default=220)
    parser.add_argument("--catastrophe-cost", type=float, default=1.0)
    parser.add_argument("--target-intervention-rate", type=float, default=0.4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--primary-alpha", type=float, default=0.1)
    args = parser.parse_args()
    calibrate(args)


if __name__ == "__main__":
    main()
