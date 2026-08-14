#!/usr/bin/env python3
"""Collect exact-state outcomes for Base, DetourComplete, and RetreatHold.

For each selected placement, a nominal rollout is captured under on-path glass,
off-path glass, and no-glass conditions.  Decision states are aligned to the
on-path catastrophe at several horizons.  Every option is then run from the
same serialized simulator/controller state and assigned exactly one terminal
label: task_success, catastrophe, or safe_noncompletion.

Raw temporal features are stored here.  PCA must be fitted later using only the
source-disjoint training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.counterfactual_router import (
    DEFAULT_HORIZONS,
    OPTIONS,
    classify_option_outcome,
    temporal_window,
    validate_decision_rows,
)
from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import array_sha256, canonical_sha256
from crashbench.predicates import build_any
from crashbench.provenance import repository_provenance, require_checkpoint_revision
from crashbench.recovery import DetourComplete, RetreatHold
from scripts.capture_glass_detector_placements import (
    _condition_glasses,
    _seed_everything,
    build_capture_plan,
)
from scripts.collect_glass_recovery_pairs import (
    CandidateRejected,
    DEFAULT_BOWL_GRASP_OFFSET,
    PLATE,
    TARGET,
    _branch_start_hashes,
    _controller_glass,
    _controller_state_sha256,
    _continuation_state_sha256,
    _glass_force,
    _glass_predicate_specs,
    _observation_sha256,
    _prime_glass_predicates,
    _require_branch_start_hashes,
)


CONDITIONS = ("glass", "offpath", "noglass")


def _scan_condition(
    env: LiberoEnv,
    policy,
    source_state: np.ndarray,
    placement,
    condition: str,
    *,
    settle_steps: int,
    max_steps: int,
) -> dict[str, Any]:
    glasses = _condition_glasses(placement, condition)
    obs = env.reset_to(source_state, movable_objects=glasses or None)
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(env.dummy_action())
    crash = None
    if glasses:
        crash = build_any(_glass_predicate_specs(glasses))
        _prime_glass_predicates(crash, env.sim_view)
    model_xml = env.model_xml()
    states: list[np.ndarray] = []
    controller_states: list[dict[str, np.ndarray]] = []
    observations: list[dict[str, np.ndarray]] = []
    rows: list[dict[str, np.ndarray | float]] = []
    collision_step = None
    succeeded = False
    peak_force = 0.0
    for step in range(max_steps):
        states.append(env.flat_state())
        controller_states.append(env.controller_state())
        observations.append({
            key: np.asarray(value).copy() for key, value in obs.items()
        })
        policy_obs = env.policy_observation(obs, policy.resize_size)
        nominal_action = np.asarray(
            policy.act(policy_obs, placement.instruction), dtype=np.float32
        )
        hidden = policy.last_hidden
        if hidden is None:
            raise RuntimeError("OpenVLA hidden hook returned None")
        row = {
            "hidden": np.asarray(hidden, dtype=np.float16),
            "robot_state": np.asarray(policy_obs["state"], dtype=np.float32),
            "nominal_action": nominal_action,
        }
        obs, _, done, _ = env.step(nominal_action.tolist())
        force = 0.0 if not glasses else _glass_force(env.sim_view, glasses)
        peak_force = max(peak_force, force)
        row["glass_force_after"] = float(force)
        rows.append(row)
        if crash is not None and crash(env.sim_view):
            collision_step = step
            break
        if done:
            succeeded = True
            break
    return {
        "condition": condition,
        "glasses": glasses,
        "model_xml": model_xml,
        "states": states,
        "controller_states": controller_states,
        "observations": observations,
        "rows": rows,
        "collision_step": collision_step,
        "crashed": collision_step is not None,
        "succeeded": succeeded,
        "peak_force_n": float(peak_force),
    }


def _expected_hashes(scan: Mapping[str, Any], anchor_index: int) -> dict[str, str]:
    state = np.asarray(scan["states"][anchor_index])
    runtime = scan["controller_states"][anchor_index]
    observation = scan["observations"][anchor_index]
    return {
        "simulator_state_sha256": array_sha256(state),
        "controller_state_sha256": _controller_state_sha256(runtime),
        "continuation_state_sha256": _continuation_state_sha256(runtime),
        "observation_sha256": _observation_sha256(observation),
        "model_xml_sha256": hashlib.sha256(
            str(scan["model_xml"]).encode("utf-8")
        ).hexdigest(),
    }


def _branch_start_catastrophic(
    env: LiberoEnv,
    scan: Mapping[str, Any],
    anchor_index: int,
    expected_hashes: Mapping[str, str],
    *,
    label: str,
) -> bool:
    """Reject anchors where catastrophe predates every branch action."""

    glasses = list(scan["glasses"])
    if not glasses:
        return False
    _restore_anchor(env, scan, anchor_index, expected_hashes, label=label)
    crash = build_any(_glass_predicate_specs(glasses))
    try:
        _prime_glass_predicates(crash, env.sim_view)
    except CandidateRejected as exc:
        if exc.reason == "invalid_initial_state":
            return True
        raise
    return False


def _restore_anchor(
    env: LiberoEnv,
    scan: Mapping[str, Any],
    anchor_index: int,
    expected_hashes: Mapping[str, str],
    *,
    label: str,
) -> dict:
    obs = env.reset_to_exact(
        np.asarray(scan["states"][anchor_index]), model_xml=str(scan["model_xml"])
    )
    obs = env.restore_controller_state(scan["controller_states"][anchor_index])
    _require_branch_start_hashes(
        _branch_start_hashes(env, obs), expected_hashes, label=label
    )
    return obs


def _run_base_continue(
    env: LiberoEnv,
    policy,
    obs: dict,
    instruction: str,
    glasses: list[dict],
    *,
    max_steps: int,
) -> dict[str, Any]:
    crash = None
    if glasses:
        crash = build_any(_glass_predicate_specs(glasses))
        _prime_glass_predicates(crash, env.sim_view)
    peak_force = 0.0
    for step in range(max_steps):
        if env.episode_terminated():
            return {
                "crashed": False, "succeeded": False, "steps": step,
                "peak_force_n": float(peak_force),
                "continuation_mode": "online_frozen_vla",
                "termination": "robosuite_episode_horizon",
            }
        policy_obs = env.policy_observation(obs, policy.resize_size)
        action = np.asarray(policy.act(policy_obs, instruction), dtype=np.float32)
        obs, _, done, _ = env.step(action.tolist())
        force = 0.0 if not glasses else _glass_force(env.sim_view, glasses)
        peak_force = max(peak_force, force)
        if crash is not None and crash(env.sim_view):
            return {
                "crashed": True, "succeeded": False, "steps": step + 1,
                "peak_force_n": float(peak_force),
                "continuation_mode": "online_frozen_vla",
            }
        if done:
            return {
                "crashed": False, "succeeded": True, "steps": step + 1,
                "peak_force_n": float(peak_force),
                "continuation_mode": "online_frozen_vla",
            }
        if env.episode_terminated():
            return {
                "crashed": False, "succeeded": False, "steps": step + 1,
                "peak_force_n": float(peak_force),
                "continuation_mode": "online_frozen_vla",
                "termination": "robosuite_episode_horizon",
            }
    return {
        "crashed": False,
        "succeeded": False,
        "steps": max_steps,
        "peak_force_n": float(peak_force),
        "continuation_mode": "online_frozen_vla",
    }


def _run_structured_option(
    env: LiberoEnv,
    obs: dict,
    glasses: list[dict],
    controller,
    *,
    max_steps: int,
) -> dict[str, Any]:
    crash = None
    if glasses:
        crash = build_any(_glass_predicate_specs(glasses))
        _prime_glass_predicates(crash, env.sim_view)
    peak_force = 0.0
    controller.engage(obs)
    for step in range(max_steps):
        if env.episode_terminated():
            return {
                "crashed": False, "succeeded": False, "steps": step,
                "peak_force_n": float(peak_force),
                "controller_final_stage": getattr(controller, "i", None),
                "termination": "robosuite_episode_horizon",
            }
        action = np.asarray(controller.step(obs), dtype=np.float32)
        obs, _, done, _ = env.step(action.tolist())
        force = 0.0 if not glasses else _glass_force(env.sim_view, glasses)
        peak_force = max(peak_force, force)
        if crash is not None and crash(env.sim_view):
            return {
                "crashed": True, "succeeded": False, "steps": step + 1,
                "peak_force_n": float(peak_force),
                "controller_final_stage": getattr(controller, "i", None),
            }
        if done:
            return {
                "crashed": False, "succeeded": True, "steps": step + 1,
                "peak_force_n": float(peak_force),
                "controller_final_stage": getattr(controller, "i", None),
            }
        if env.episode_terminated():
            return {
                "crashed": False, "succeeded": False, "steps": step + 1,
                "peak_force_n": float(peak_force),
                "controller_final_stage": getattr(controller, "i", None),
                "termination": "robosuite_episode_horizon",
            }
    return {
        "crashed": False, "succeeded": False, "steps": max_steps,
        "peak_force_n": float(peak_force),
        "controller_final_stage": getattr(controller, "i", None),
    }


def _detour_controller(obs: Mapping[str, Any], glass: dict, args: argparse.Namespace):
    bowl = np.asarray(obs[f"{TARGET}_pos"], dtype=float)
    plate = np.asarray(obs[f"{PLATE}_pos"], dtype=float)
    return DetourComplete(
        _controller_glass(glass), bowl, plate,
        side=args.detour_side,
        lane_margin=args.detour_lane_margin,
        transit_z=float(bowl[2] + args.detour_lift_offset),
        descend_off=args.detour_descend_offset,
        leg_cap=args.detour_leg_cap,
        target_name=TARGET,
        orientation_target=None,
        path_aligned=True,
        grasp_xy_offset=args.detour_grasp_xy_offset,
        departure_clearance=args.detour_departure_clearance,
    )


def _apply_frozen_detour_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.detour_config is None:
        return None
    path = Path(args.detour_config).resolve()
    config = json.loads(path.read_text())
    required = {
        "side", "lane_margin", "lift_offset", "descend_offset",
        "grasp_xy_offset", "departure_clearance", "orientation_target",
        "path_aligned",
    }
    if set(config) != required:
        raise ValueError(
            f"frozen detour config keys must be exactly {sorted(required)}; "
            f"got {sorted(config)}"
        )
    if config["orientation_target"] is not None or config["path_aligned"] is not True:
        raise ValueError("counterfactual Detour requires null orientation and path alignment")
    grasp_xy = [float(value) for value in config["grasp_xy_offset"]]
    if len(grasp_xy) != 2:
        raise ValueError("frozen detour grasp_xy_offset must contain two values")
    args.detour_side = float(config["side"])
    args.detour_lane_margin = float(config["lane_margin"])
    args.detour_lift_offset = float(config["lift_offset"])
    args.detour_descend_offset = float(config["descend_offset"])
    args.detour_grasp_xy_offset = grasp_xy
    args.detour_departure_clearance = float(config["departure_clearance"])
    return config


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def collect(args: argparse.Namespace) -> dict[str, Any]:
    frozen_detour_config = _apply_frozen_detour_config(args)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    revision = require_checkpoint_revision(args.checkpoint_revision)
    repo = repository_provenance(Path(__file__).resolve().parents[1], require_clean=True)
    declared = os.environ.get("CB_CODE_COMMIT")
    if declared is not None and declared != repo["git_commit"]:
        raise SystemExit("CB_CODE_COMMIT does not match checked-out source")
    plan, source_splits = build_capture_plan(
        args.exposed_placements, args.development_placements,
        placement_keys=None if not args.placement_key else set(args.placement_key),
    )
    selected_splits = set(args.detector_splits)
    plan = [item for item in plan if item.detector_split in selected_splits]
    if args.max_placements is not None:
        plan = plan[:args.max_placements]
    if not plan:
        raise ValueError("selected counterfactual capture plan is empty")
    source_splits = {
        item.placement.source_state_sha256: item.detector_split for item in plan
    }

    from crashbench.policies import OpenVLAPolicy
    policy = OpenVLAPolicy(
        pretrained_checkpoint=args.checkpoint,
        checkpoint_revision=revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        capture_hidden=True,
    )
    envs: dict[tuple[str, int], LiberoEnv] = {}
    decision_metadata: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []
    hidden_windows: list[np.ndarray] = []
    robot_windows: list[np.ndarray] = []
    action_windows: list[np.ndarray] = []
    history_masks: list[np.ndarray] = []
    exclusions: list[dict[str, Any]] = []
    attempted_placements = 0
    valid_placements = 0

    for item in plan:
        if (
            args.target_valid_placements is not None
            and valid_placements >= args.target_valid_placements
        ):
            break
        attempted_placements += 1
        placement = item.placement
        source_path = item.manifest_path.parent / placement.source_state_path
        source_state = np.load(source_path, allow_pickle=False)
        if array_sha256(source_state) != placement.source_state_sha256:
            raise ValueError(f"{item.key}: source-state hash mismatch")
        env_key = (str(placement.task_suite), int(placement.task_id))
        if env_key not in envs:
            envs[env_key] = LiberoEnv(*env_key, seed=args.rollout_seed)
        env = envs[env_key]
        scans = {}
        try:
            for condition in CONDITIONS:
                _seed_everything(args.rollout_seed)
                env.seed(args.rollout_seed)
                policy.reset()
                scans[condition] = _scan_condition(
                    env, policy, source_state, placement, condition,
                    settle_steps=args.settle_steps, max_steps=args.scan_steps,
                )
        except CandidateRejected as exc:
            exclusion = {
                "placement_key": item.key,
                "source_state_sha256": placement.source_state_sha256,
                "reason": f"scan_{exc.reason}",
                "detail": str(exc),
            }
            exclusions.append(exclusion)
            _append_jsonl(output / "capture_progress.jsonl", exclusion)
            continue
        glass_scan = scans["glass"]
        if not glass_scan["crashed"]:
            exclusion = {
                "placement_key": item.key,
                "source_state_sha256": placement.source_state_sha256,
                "reason": "onpath_no_catastrophe",
            }
            exclusions.append(exclusion)
            _append_jsonl(output / "capture_progress.jsonl", exclusion)
            continue
        collision_step = int(glass_scan["collision_step"])
        decisions_before = len(decision_metadata)
        for horizon in args.horizons:
            anchor_index = collision_step - int(horizon) + 1
            if anchor_index < 0:
                exclusions.append({
                    "placement_key": item.key,
                    "source_state_sha256": placement.source_state_sha256,
                    "horizon_actions": int(horizon),
                    "reason": "catastrophe_before_horizon",
                })
                continue
            for condition in CONDITIONS:
                scan = scans[condition]
                if anchor_index >= len(scan["states"]):
                    exclusions.append({
                        "placement_key": item.key,
                        "source_state_sha256": placement.source_state_sha256,
                        "horizon_actions": int(horizon),
                        "condition": condition,
                        "reason": "condition_terminal_before_matched_anchor",
                    })
                    continue
                expected = _expected_hashes(scan, anchor_index)
                if _branch_start_catastrophic(
                    env, scan, anchor_index, expected,
                    label=f"{item.key}:{condition}:H{horizon}:preflight",
                ):
                    exclusion = {
                        "placement_key": item.key,
                        "source_state_sha256": placement.source_state_sha256,
                        "horizon_actions": int(horizon),
                        "condition": condition,
                        "reason": "branch_start_catastrophic",
                    }
                    exclusions.append(exclusion)
                    _append_jsonl(output / "capture_progress.jsonl", exclusion)
                    continue
                feature_index = len(decision_metadata)
                hidden, history_mask = temporal_window(
                    [row["hidden"] for row in scan["rows"]],
                    anchor_index, args.history_length,
                )
                robot, robot_mask = temporal_window(
                    [row["robot_state"] for row in scan["rows"]],
                    anchor_index, args.history_length,
                )
                action, action_mask = temporal_window(
                    [row["nominal_action"] for row in scan["rows"]],
                    anchor_index, args.history_length,
                )
                if not (
                    np.array_equal(history_mask, robot_mask)
                    and np.array_equal(history_mask, action_mask)
                ):
                    raise RuntimeError("temporal feature masks disagree")
                decision_id = canonical_sha256({
                    "source_state_sha256": placement.source_state_sha256,
                    "placement_id": placement.placement_id,
                    "condition": condition,
                    "horizon_actions": int(horizon),
                    "anchor_state_sha256": expected["simulator_state_sha256"],
                })
                hidden_windows.append(hidden)
                robot_windows.append(robot)
                action_windows.append(action)
                history_masks.append(history_mask)
                decision_metadata.append({
                    "decision_id": decision_id,
                    "feature_index": feature_index,
                    "placement_key": item.key,
                    "placement_id": placement.placement_id,
                    "source_state_sha256": placement.source_state_sha256,
                    "split": item.detector_split,
                    "condition": condition,
                    "horizon_actions": int(horizon),
                    "matched_scan_index": anchor_index,
                    "onpath_collision_step": collision_step,
                    "history_valid_steps": int(history_mask.sum()),
                    "branch_start_hashes": expected,
                })

                for option in OPTIONS:
                    obs = _restore_anchor(
                        env, scan, anchor_index, expected,
                        label=f"{decision_id}:{option}",
                    )
                    if option == "base_continue":
                        policy.reset()
                        result = _run_base_continue(
                            env, policy, obs, placement.instruction,
                            list(scan["glasses"]), max_steps=args.base_steps,
                        )
                    elif option == "detour_complete":
                        controller_glass = (
                            placement.on_path_glass if condition == "noglass"
                            else scan["glasses"][0]
                        )
                        result = _run_structured_option(
                            env, obs, list(scan["glasses"]),
                            _detour_controller(obs, controller_glass, args),
                            max_steps=args.detour_steps,
                        )
                    else:
                        result = _run_structured_option(
                            env, obs, list(scan["glasses"]),
                            RetreatHold(back=args.retreat_back, up=args.retreat_up),
                            max_steps=args.retreat_steps,
                        )
                    outcome = classify_option_outcome(
                        crashed=bool(result["crashed"]),
                        succeeded=bool(result["succeeded"]),
                    )
                    option_row = {
                        "schema_version": 1,
                        "decision_id": decision_id,
                        "feature_index": feature_index,
                        "placement_key": item.key,
                        "placement_id": placement.placement_id,
                        "source_state_sha256": placement.source_state_sha256,
                        "split": item.detector_split,
                        "condition": condition,
                        "horizon_actions": int(horizon),
                        "option": option,
                        "outcome": outcome,
                        **result,
                    }
                    option_rows.append(option_row)
                    _append_jsonl(output / "option_rollouts.jsonl", option_row)
                _append_jsonl(output / "capture_progress.jsonl", {
                    "event": "decision_complete",
                    **decision_metadata[-1],
                })
        if len(decision_metadata) > decisions_before:
            valid_placements += 1

    validation = validate_decision_rows(option_rows)
    feature_path = output / "decision_features.npz"
    np.savez_compressed(
        feature_path,
        hidden=np.asarray(hidden_windows, dtype=np.float16),
        robot_state=np.asarray(robot_windows, dtype=np.float32),
        nominal_action=np.asarray(action_windows, dtype=np.float32),
        history_mask=np.asarray(history_masks, dtype=np.float32),
    )
    (output / "decision_metadata.json").write_text(
        json.dumps(decision_metadata, indent=2, sort_keys=True) + "\n"
    )
    protocol = {
        "horizons": list(args.horizons),
        "history_length": args.history_length,
        "conditions": list(CONDITIONS),
        "options": list(OPTIONS),
        "base": "online frozen-OpenVLA continuation from the exact restored state",
        "detour": {
            "side": args.detour_side,
            "lane_margin": args.detour_lane_margin,
            "lift_offset": args.detour_lift_offset,
            "descend_offset": args.detour_descend_offset,
            "leg_cap": args.detour_leg_cap,
            "departure_clearance": args.detour_departure_clearance,
            "grasp_xy_offset": list(args.detour_grasp_xy_offset),
            "path_aligned": True,
            "privileged_geometry": True,
            "frozen_config": frozen_detour_config,
        },
        "retreat": {"back": args.retreat_back, "up": args.retreat_up},
        "terminal_outcomes": [
            "task_success", "catastrophe", "safe_noncompletion"
        ],
    }
    manifest = {
        "schema_version": 1,
        "kind": "counterfactual_option_rollout_capture",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repo,
        "checkpoint_identity": policy.checkpoint_identity,
        "checkpoint_revision": revision,
        "rollout_seed": args.rollout_seed,
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "source_splits": source_splits,
        "plan_placements": len(plan),
        "attempted_placements": attempted_placements,
        "valid_placements": valid_placements,
        "validation": validation,
        "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
        "exclusions": exclusions,
        "artifacts": {
            "features": feature_path.name,
            "decision_metadata": "decision_metadata.json",
            "option_rollouts": "option_rollouts.jsonl",
        },
        "feature_contract": {
            "hidden": "raw frozen-VLA hidden; fit PCA on train source states only",
            "robot_state_dim": 8,
            "nominal_action_dim": 7,
            "causal_history": True,
        },
    }
    (output / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposed-placements", required=True)
    parser.add_argument("--development-placements", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial"
    )
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--scan-steps", type=int, default=220)
    parser.add_argument("--base-steps", type=int, default=220)
    parser.add_argument("--detour-steps", type=int, default=900)
    parser.add_argument("--retreat-steps", type=int, default=80)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS)
    )
    parser.add_argument(
        "--detector-splits", nargs="+",
        choices=("train", "calibration", "development"),
        default=["train", "calibration", "development"],
    )
    parser.add_argument("--placement-key", action="append")
    parser.add_argument("--max-placements", type=int)
    parser.add_argument("--target-valid-placements", type=int)
    parser.add_argument("--detour-side", type=float, default=1.0)
    parser.add_argument("--detour-lane-margin", type=float, default=0.12)
    parser.add_argument("--detour-lift-offset", type=float, default=0.30)
    parser.add_argument("--detour-descend-offset", type=float, default=0.018)
    parser.add_argument("--detour-leg-cap", type=int, default=140)
    parser.add_argument("--detour-departure-clearance", type=float, default=0.06)
    parser.add_argument(
        "--detour-grasp-xy-offset", type=float, nargs=2,
        default=list(DEFAULT_BOWL_GRASP_OFFSET[:2]),
    )
    parser.add_argument(
        "--detour-config",
        help="frozen_detour_config.json emitted by the development-only sweep",
    )
    parser.add_argument("--retreat-back", type=float, default=0.14)
    parser.add_argument("--retreat-up", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.history_length < 1
        or not args.horizons
        or len(set(args.horizons)) != len(args.horizons)
        or any(horizon < 1 for horizon in args.horizons)
        or args.max_placements is not None and args.max_placements < 1
        or args.target_valid_placements is not None
        and args.target_valid_placements < 1
    ):
        raise SystemExit("history, horizons, and max placements must be positive")
    collect(args)


if __name__ == "__main__":
    main()
