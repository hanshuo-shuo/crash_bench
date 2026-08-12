#!/usr/bin/env python3
"""Run the pre-Pilot-B continuation x oracle 2x2 diagnostic.

This script is deliberately diagnostic-only.  It may be run on the exposed r7
placements to locate a technical defect, but its outcomes must never admit
those placements into a training/evaluation cohort.

Rows compare two ways of returning to the captured exact-H anchor:

``in_memory``
    Rewind the existing MjModel/MjData and restore the runtime snapshot.
``serialized``
    Rebuild from the exact captured model XML, then restore the same snapshot.

Columns replay the captured Base suffix or run the searched oracle recovery.
The live, never-restored Base continuation is retained as the reference event.
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
from typing import Any, Callable

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import (
    array_sha256,
    exact_h_anchor_index,
    read_placement_manifest,
)
from crashbench.policies import OpenVLAPolicy
from scripts.collect_glass_recovery_pairs import (
    TARGET,
    _branch_start_hashes,
    _continuation_state_sha256,
    _controller_state_sha256,
    _observation_sha256,
    _replay_nominal_actions,
    _require_branch_start_hashes,
    _roll_nominal,
    _run_offpath,
    _search_oracle,
    _seed_rollout,
)


def _captured_hashes(
    env: LiberoEnv,
    state: np.ndarray,
    runtime: dict[str, np.ndarray],
    observation: dict[str, np.ndarray],
    model_xml: str,
) -> dict[str, str]:
    return {
        "simulator_state_sha256": array_sha256(state),
        "controller_state_sha256": _controller_state_sha256(runtime),
        "continuation_state_sha256": _continuation_state_sha256(runtime),
        "observation_sha256": _observation_sha256(observation),
        "model_xml_sha256": hashlib.sha256(model_xml.encode("utf-8")).hexdigest(),
    }


def _runtime_differences(
    expected: dict[str, np.ndarray], actual: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Describe exact per-field snapshot drift without embedding large arrays."""

    differences: dict[str, Any] = {}
    for key in sorted(set(expected) | set(actual)):
        if key not in expected:
            differences[key] = {"kind": "unexpected"}
            continue
        if key not in actual:
            differences[key] = {"kind": "missing"}
            continue
        left, right = np.asarray(expected[key]), np.asarray(actual[key])
        if array_sha256(left) == array_sha256(right):
            continue
        detail: dict[str, Any] = {
            "kind": "value_mismatch",
            "expected_dtype": str(left.dtype),
            "actual_dtype": str(right.dtype),
            "expected_shape": list(left.shape),
            "actual_shape": list(right.shape),
            "expected_sha256": array_sha256(left),
            "actual_sha256": array_sha256(right),
        }
        if left.shape == right.shape and left.dtype.kind in "biufc" and right.dtype.kind in "biufc":
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            detail["max_abs_delta"] = float(np.max(delta)) if delta.size else 0.0
        differences[key] = detail
    return differences


def _restore_cell(
    reset: Callable[[], dict],
    env: LiberoEnv,
    expected: dict[str, str],
    expected_runtime: dict[str, np.ndarray],
    *,
    label: str,
) -> tuple[dict | None, dict[str, Any]]:
    try:
        obs = reset()
        actual = _branch_start_hashes(env, obs)
        _require_branch_start_hashes(actual, expected, label=label)
    except Exception as exc:
        return None, {
            "start_exact": False,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_differences": _runtime_differences(
                expected_runtime, env.controller_state()
            ),
        }
    return obs, {"start_exact": True, "start_hashes": actual}


def _base_cell(
    reset: Callable[[], dict],
    env: LiberoEnv,
    expected: dict[str, str],
    expected_runtime: dict[str, np.ndarray],
    glass: dict,
    suffix: list[dict],
    *,
    label: str,
) -> dict[str, Any]:
    obs, cell = _restore_cell(
        reset, env, expected, expected_runtime, label=label
    )
    if obs is None:
        return cell
    try:
        replay = _replay_nominal_actions(env, obs, [glass], suffix)
    except Exception as exc:
        return {**cell, "error": f"{type(exc).__name__}: {exc}"}
    expected_step = len(suffix) - 1
    return {
        **cell,
        "crashed": bool(replay["crashed"]),
        "expected_collision_step": expected_step,
        "actual_collision_step": replay["collision_step"],
        "exact_h_replay": bool(
            replay["crashed"] and replay["collision_step"] == expected_step
        ),
        "peak_glass_force_n": round(float(replay["peak_force"]), 5),
    }


def _oracle_cell(
    reset: Callable[[], dict],
    env: LiberoEnv,
    expected: dict[str, str],
    expected_runtime: dict[str, np.ndarray],
    policy,
    placement,
    *,
    oracle_steps: int,
    counterfactual_peak_force: float,
    orientation_targets: list[list[float] | None] | None,
    control_grasp_offset: list[float] | None,
    label: str,
) -> dict[str, Any]:
    # Fail before search if even one reset cannot recover the captured anchor.
    obs, cell = _restore_cell(
        reset, env, expected, expected_runtime, label=label
    )
    if obs is None:
        return cell
    try:
        oracle, attempts = _search_oracle(
            reset,
            env,
            policy,
            placement,
            [placement.on_path_glass],
            oracle_steps,
            collect_success=True,
            counterfactual_peak_force=counterfactual_peak_force,
            orientation_targets=orientation_targets,
            control_grasp_offset=control_grasp_offset,
            capture_success_rows=False,
        )
    except Exception as exc:
        return {**cell, "error": f"{type(exc).__name__}: {exc}"}
    selected = None if oracle is None else oracle.get("config")
    best_failure = None
    if oracle is None and attempts:
        best = max(attempts, key=lambda row: (
            int(row["controller_final_stage"]),
            float(row.get("max_target_z_m") or -1.0),
            -float(row.get("final_bowl_plate_xy_m") or 99.0),
        ))
        best_failure = {
            key: best.get(key) for key in (
                "attempt_index", "controller_final_stage", "crashed", "steps",
                "initial_target_z_m", "max_target_z_m",
                "final_bowl_plate_xy_m", "final_bowl_z_m", "side",
                "lane_margin", "transit_z", "descend_off",
                "orientation_target", "grasp_xy_offset",
            )
        }
    return {
        **cell,
        "safe_task_success": oracle is not None,
        "search_attempts": len(attempts),
        "collision_attempts": sum(bool(row["crashed"]) for row in attempts),
        "timeout_or_task_failure_attempts": sum(
            not bool(row["crashed"]) and not bool(row["succeeded"])
            for row in attempts
        ),
        "furthest_controller_stage": max(
            (int(row["controller_final_stage"]) for row in attempts), default=-1
        ),
        "selected_config": selected,
        "selected_steps": None if oracle is None else int(oracle["steps"]),
        "best_failure": best_failure,
    }


def _control_probe(
    env: LiberoEnv,
    policy,
    placement,
    matched_robot: np.ndarray,
    runtime: dict[str, np.ndarray],
    *,
    control_steps: int,
) -> tuple[list[list[float] | None] | None, list[float] | None, dict[str, Any]]:
    obs = env.reset_to(matched_robot, movable_objects=[placement.off_path_glass])
    obs = env.restore_controller_state(runtime, restore_observables=False)
    result = _run_offpath(env, policy, obs, placement, control_steps)
    diagnostic = {
        "crashed": bool(result["crashed"]),
        "succeeded": bool(result["succeeded"]),
        "steps": int(result["steps"]),
    }
    if result["crashed"] or not result["rows"]:
        return None, None, diagnostic
    baseline = np.asarray(placement.metadata["bowl_xyz"], dtype=float)
    _, closest = min(
        enumerate(result["rows"]),
        key=lambda item: float(np.linalg.norm(
            np.asarray(item[1]["robot_state"], dtype=float)[:3] - baseline
        )),
    )
    robot_state = np.asarray(closest["robot_state"], dtype=float)
    orientation = robot_state[3:6].round(7).tolist()
    offset = (robot_state[:3] - baseline).round(7).tolist()
    diagnostic.update(
        control_reach_orientation=orientation,
        control_grasp_offset_xyz=offset,
        pose_usable=True,
        pose_source=(
            "successful_control" if result["succeeded"]
            else "closest_collision_free_control_approach"
        ),
    )
    return [orientation, None], offset, diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument(
        "--placement-id",
        help="run only one named diagnostic placement",
    )
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--scan-steps", type=int, default=220)
    parser.add_argument("--control-steps", type=int, default=220)
    parser.add_argument("--oracle-steps", type=int, default=220)
    args = parser.parse_args()
    if args.horizon < 1 or args.max_candidates < 1:
        raise SystemExit("horizon and max-candidates must be positive")

    placements, _ = read_placement_manifest(args.placements)
    if args.placement_id:
        selected = [
            placement for placement in placements
            if placement.placement_id == args.placement_id
        ]
        if not selected:
            raise SystemExit(f"unknown placement id: {args.placement_id}")
    else:
        selected = placements[: args.max_candidates]
    placement_root = Path(args.placements).resolve().parent
    policy = OpenVLAPolicy(
        pretrained_checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        capture_hidden=True,
    )
    envs: dict[tuple[str, int], tuple[LiberoEnv, LiberoEnv]] = {}
    rows: list[dict[str, Any]] = []

    for placement in selected:
        key = (placement.task_suite, placement.task_id)
        if key not in envs:
            envs[key] = (
                LiberoEnv(*key, seed=args.rollout_seed),
                LiberoEnv(*key, seed=args.rollout_seed),
            )
        memory_env, serialized_env = envs[key]
        _seed_rollout(args.rollout_seed)
        policy.reset()
        source = np.load(placement_root / placement.source_state_path, allow_pickle=False)
        obs = memory_env.reset_to(source, movable_objects=[placement.on_path_glass])
        for _ in range(args.settle_steps):
            obs, _, _, _ = memory_env.step(memory_env.dummy_action())
        scan = _roll_nominal(
            memory_env,
            policy,
            obs,
            placement.instruction,
            [placement.on_path_glass],
            args.scan_steps,
            capture_states=True,
            capture_rows=True,
        )
        row: dict[str, Any] = {
            "placement_id": placement.placement_id,
            "split": placement.split,
            "horizon_actions": args.horizon,
            "live_base_crash": bool(scan["crashed"]),
            "live_collision_step": scan.get("collision_step"),
        }
        if not scan["crashed"]:
            row["status"] = "no_live_base_crash"
            rows.append(row)
            continue
        try:
            anchor_index = exact_h_anchor_index(
                int(scan["collision_step"]), args.horizon
            )
        except ValueError as exc:
            row.update(status="no_exact_h_anchor", error=str(exc))
            rows.append(row)
            continue

        anchor = np.asarray(scan["states"][anchor_index], dtype=np.float64)
        runtime = scan["controller_states"][anchor_index]
        captured_obs = scan["observations"][anchor_index]
        model_xml = str(scan["model_xml"])
        suffix = scan["rows"][anchor_index:int(scan["collision_step"]) + 1]
        expected = _captured_hashes(
            memory_env, anchor, runtime, captured_obs, model_xml
        )
        model = memory_env._raw_model()
        matched_robot = memory_env._strip_movable_state(
            anchor, int(model.nq), int(model.nv), 1
        )
        try:
            orientations, grasp_offset, control = _control_probe(
                serialized_env,
                policy,
                placement,
                matched_robot,
                runtime,
                control_steps=args.control_steps,
            )
        except Exception as exc:
            orientations, grasp_offset = None, None
            control = {"error": f"{type(exc).__name__}: {exc}"}

        def reset_memory():
            return memory_env.restore_to_exact_in_place(anchor, runtime)

        def reset_serialized():
            restored = serialized_env.reset_to_exact(anchor, model_xml=model_xml)
            return serialized_env.restore_controller_state(runtime)

        row.update(
            status="complete",
            anchor_index=anchor_index,
            captured_start_hashes=expected,
            control_probe=control,
            in_memory={
                "base_suffix": _base_cell(
                    reset_memory, memory_env, expected, runtime,
                    placement.on_path_glass,
                    suffix, label=f"{placement.placement_id}/in-memory/base",
                ),
                "oracle_recovery": _oracle_cell(
                    reset_memory, memory_env, expected, runtime, policy, placement,
                    oracle_steps=args.oracle_steps,
                    counterfactual_peak_force=float(scan["peak_force"]),
                    orientation_targets=orientations,
                    control_grasp_offset=grasp_offset,
                    label=f"{placement.placement_id}/in-memory/oracle",
                ),
            },
            serialized={
                "base_suffix": _base_cell(
                    reset_serialized, serialized_env, expected, runtime,
                    placement.on_path_glass, suffix,
                    label=f"{placement.placement_id}/serialized/base",
                ),
                "oracle_recovery": _oracle_cell(
                    reset_serialized, serialized_env, expected, runtime,
                    policy, placement,
                    oracle_steps=args.oracle_steps,
                    counterfactual_peak_force=float(scan["peak_force"]),
                    orientation_targets=orientations,
                    control_grasp_offset=grasp_offset,
                    label=f"{placement.placement_id}/serialized/oracle",
                ),
            },
        )
        rows.append(row)
        print(f"DIAG {placement.placement_id} complete", flush=True)

    completed = [row for row in rows if row.get("status") == "complete"]
    aggregate: dict[str, Any] = {
        "candidates": len(rows),
        "live_base_crashes": sum(bool(row["live_base_crash"]) for row in rows),
        "complete_exact_h_diagnostics": len(completed),
    }
    for reset_kind in ("in_memory", "serialized"):
        aggregate[reset_kind] = {
            "exact_starts": sum(
                bool(row[reset_kind][cell].get("start_exact"))
                for row in completed for cell in ("base_suffix", "oracle_recovery")
            ),
            "exact_h_base_replays": sum(
                bool(row[reset_kind]["base_suffix"].get("exact_h_replay"))
                for row in completed
            ),
            "oracle_safe_task_successes": sum(
                bool(row[reset_kind]["oracle_recovery"].get("safe_task_success"))
                for row in completed
            ),
            "start_errors": dict(Counter(
                row[reset_kind][cell].get("error", "")
                for row in completed for cell in ("base_suffix", "oracle_recovery")
                if row[reset_kind][cell].get("error")
            )),
        }
    payload = {
        "schema_version": 1,
        "kind": "glass_continuation_oracle_2x2_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "cohort_admission_allowed": False,
        "placements": str(Path(args.placements).resolve()),
        "checkpoint_revision": args.checkpoint_revision,
        "rollout_seed": args.rollout_seed,
        "horizon_actions": args.horizon,
        "aggregate": aggregate,
        "rows": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
