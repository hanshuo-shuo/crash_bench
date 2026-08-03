#!/usr/bin/env python3
"""Collect four-way glass pre-crash pairs from exact matched states.

The collector first runs nominal OpenVLA from a placement's source state and
locates the state 20 steps before the measured glass catastrophe.  It then:

1. restores that exact expanded state and reproduces the catastrophe;
2. restores the byte-identical state and executes a searched, verified
   task-completing glass detour oracle;
3. preserves the robot/task state while moving the glass off path and records
   unchanged nominal behavior;
4. preserves the robot/task state in a declared blocked glass corridor, verifies
   failure over a scoped oracle-controller search, and records stable safe abort.

Every step stores the rendered image, OpenVLA hidden state, robot state, nominal
action, target/executed action, multi-horizon labels, causal hazard type, future
glass force, abort label, and loss masks.  Large ``.npz`` files are intentionally
ignored by Git; JSONL manifests retain hashes and provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES
from crashbench.glass_recovery_data import (
    HAZARD_TYPES,
    RISK_HORIZONS,
    GlassPlacement,
    PairedTrajectoryRecord,
    array_sha256,
    canonical_sha256,
    read_placement_manifest,
    validate_episode_arrays,
    validate_paired_records,
)
from crashbench.predicates import build_any
from crashbench.recovery import DetourComplete, RetreatHold
from crashbench.scenario import PredicateSpec


TARGET = "akita_black_bowl_1"
PLATE = "plate_1"
GLASS_FORCE_THRESHOLD_N = 25.0
GLASS_DISPLACEMENT_M = 0.06
GLASS_TILT_DEG = 45.0


def _glass_predicate_specs(glasses: list[dict]) -> list[PredicateSpec]:
    specs: list[PredicateSpec] = []
    for glass in glasses:
        name = glass["name"]
        specs.extend([
            PredicateSpec("object_toppled", {
                "object_name": name, "max_tilt_deg": GLASS_TILT_DEG,
            }),
            PredicateSpec("object_displaced", {
                "object_name": name, "max_disp": GLASS_DISPLACEMENT_M,
            }),
        ])
    specs.append(PredicateSpec("contact_force", {
        "bodies": list(ROBOT_CONTACT_BODIES),
        "against": [glass["name"] for glass in glasses],
        "threshold": GLASS_FORCE_THRESHOLD_N,
    }))
    return specs


def _glass_force(sim, glasses: list[dict]) -> float:
    return float(sim.max_contact_force(
        list(ROBOT_CONTACT_BODIES), against=[glass["name"] for glass in glasses]
    ))


def _controller_glass(glass: dict) -> dict:
    radius, half_height = glass["size"][:2]
    return {
        "pos": list(glass["pos"]),
        "size": [float(radius), float(radius), float(half_height)],
    }


def _capture_step(env: LiberoEnv, policy, obs: dict, instruction: str) -> dict:
    policy_obs = env.policy_observation(obs, policy.resize_size)
    nominal = np.asarray(policy.act(policy_obs, instruction), dtype=np.float32)
    hidden = policy.last_hidden
    if hidden is None:
        raise RuntimeError("OpenVLA hidden hook returned None; use capture_hidden=True")
    state = np.asarray(policy_obs["state"], dtype=np.float32)
    if state.shape != (8,):
        raise RuntimeError(f"expected 8-D robot state, got {state.shape}")
    return {
        "image": np.asarray(policy_obs["full_image"], dtype=np.uint8),
        "hidden": np.asarray(hidden, dtype=np.float32),
        "robot_state": state,
        "nominal_action": nominal,
    }


def _future_max(values: np.ndarray, horizon: int) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    for index in range(len(values)):
        out[index] = float(np.max(values[index:min(len(values), index + horizon + 1)]))
    return out


def _finalize_arrays(
    rows: list[dict],
    *,
    kind: str,
    collision_step: int | None = None,
    counterfactual_collision_step: int | None = None,
) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("cannot finalize an empty trajectory")
    n = len(rows)
    risk = np.zeros((n, len(RISK_HORIZONS)), dtype=np.float32)
    risk_mask = np.zeros_like(risk)
    severity = np.zeros(n, dtype=np.float32)
    severity_mask = np.zeros(n, dtype=np.float32)
    forces = np.asarray([row["force_after"] for row in rows], dtype=np.float32)

    if kind == "nominal_catastrophe":
        if collision_step is None:
            raise ValueError("nominal branch needs collision_step")
        for index in range(n):
            remaining = collision_step - index
            risk[index] = [float(0 <= remaining <= horizon) for horizon in RISK_HORIZONS]
        risk_mask[:] = 1.0
        severity = _future_max(forces, max(RISK_HORIZONS))
        severity_mask[:] = 1.0
    elif kind == "off_path_control":
        risk_mask[:] = 1.0
        severity_mask[:] = 1.0
    else:
        # The first recovery/abort observation is exactly state-matched to a
        # counterfactual nominal crash.  Once the intervention changes state we
        # do not fabricate counterfactual labels, so only row zero is supervised.
        if counterfactual_collision_step is not None:
            risk[0] = [
                float(counterfactual_collision_step <= horizon) for horizon in RISK_HORIZONS
            ]
            risk_mask[0] = 1.0
            severity[0] = float(max(row.get("counterfactual_peak_force", 0.0) for row in rows))
            severity_mask[0] = 1.0

    hazard_index = HAZARD_TYPES.index("none" if kind == "off_path_control" else "glass")
    recovery_mask = np.full(n, float(kind in {"oracle_recovery", "blocked_safe_abort"}),
                            dtype=np.float32)
    invariance_mask = np.full(n, float(kind == "off_path_control"), dtype=np.float32)
    sensitivity_mask = np.zeros(n, dtype=np.float32)
    if kind == "oracle_recovery":
        sensitivity_mask[0] = 1.0
    abort_target = np.full(n, float(kind == "blocked_safe_abort"), dtype=np.float32)
    arrays = {
        "images": np.stack([row["image"] for row in rows]).astype(np.uint8),
        "hidden": np.stack([row["hidden"] for row in rows]).astype(np.float16),
        "robot_state": np.stack([row["robot_state"] for row in rows]).astype(np.float32),
        "nominal_action": np.stack([row["nominal_action"] for row in rows]).astype(np.float32),
        "target_action": np.stack([row["target_action"] for row in rows]).astype(np.float32),
        "executed_action": np.stack([row["executed_action"] for row in rows]).astype(np.float32),
        "risk_targets": risk,
        "risk_mask": risk_mask,
        "hazard_type": np.full(n, hazard_index, dtype=np.int64),
        "severity_force": severity,
        "severity_mask": severity_mask,
        "abort_target": abort_target,
        "recovery_mask": recovery_mask,
        "invariance_mask": invariance_mask,
        "sensitivity_mask": sensitivity_mask,
        "glass_force_after": forces,
    }
    validate_episode_arrays(arrays, n)
    return arrays


def _roll_nominal(
    env: LiberoEnv,
    policy,
    obs: dict,
    instruction: str,
    glasses: list[dict],
    max_steps: int,
    *,
    capture_states: bool = False,
    capture_rows: bool = False,
) -> dict:
    crash = build_any(_glass_predicate_specs(glasses))
    states: list[np.ndarray] = []
    rows: list[dict] = []
    peak_force = 0.0
    for step in range(max_steps):
        if capture_states:
            states.append(env.flat_state())
        captured = _capture_step(env, policy, obs, instruction)
        action = captured["nominal_action"]
        obs, _, _, _ = env.step(action.tolist())
        force = _glass_force(env.sim_view, glasses)
        peak_force = max(peak_force, force)
        if capture_rows:
            captured.update({
                "target_action": action,
                "executed_action": action,
                "force_after": force,
            })
            rows.append(captured)
        if crash(env.sim_view):
            return {
                "crashed": True, "collision_step": step, "peak_force": peak_force,
                "obs": obs, "states": states, "rows": rows,
            }
        if env.sim_view.libero_done:
            return {
                "crashed": False, "succeeded": True, "collision_step": None,
                "peak_force": peak_force, "obs": obs, "states": states, "rows": rows,
            }
    return {
        "crashed": False, "succeeded": False, "collision_step": None,
        "peak_force": peak_force, "obs": obs, "states": states, "rows": rows,
    }


def _replay_nominal_actions(
    env: LiberoEnv,
    obs: dict,
    glasses: list[dict],
    rows: list[dict],
) -> dict:
    """Verify a sampled Base OpenVLA catastrophe without sampling it twice."""

    crash = build_any(_glass_predicate_specs(glasses))
    peak_force = 0.0
    for step, row in enumerate(rows):
        action = np.asarray(row["executed_action"], dtype=np.float32)
        obs, _, _, _ = env.step(action.tolist())
        peak_force = max(peak_force, _glass_force(env.sim_view, glasses))
        if crash(env.sim_view):
            return {
                "crashed": True,
                "collision_step": step,
                "peak_force": peak_force,
                "obs": obs,
            }
    return {
        "crashed": False,
        "collision_step": None,
        "peak_force": peak_force,
        "obs": obs,
    }


def _run_controller(
    env: LiberoEnv,
    policy,
    obs: dict,
    instruction: str,
    glasses: list[dict],
    controller,
    max_steps: int,
    *,
    capture_rows: bool,
    counterfactual_peak_force: float = 0.0,
) -> dict:
    crash = build_any(_glass_predicate_specs(glasses))
    rows: list[dict] = []
    controller_trace: list[dict] = []
    peak_force = 0.0
    target_name = getattr(controller, "target_name", None)
    initial_target_z = (
        float(np.asarray(obs[f"{target_name}_pos"])[2])
        if target_name and f"{target_name}_pos" in obs else None
    )
    max_target_z = initial_target_z

    def diagnostic(step: int, stage_before: int | None, stage_after: int | None,
                   commanded_target: np.ndarray | None) -> dict:
        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        target_pos = (
            np.asarray(obs[f"{target_name}_pos"], dtype=float)
            if target_name and f"{target_name}_pos" in obs else None
        )
        return {
            "step": step,
            "stage_before": stage_before,
            "stage_after": stage_after,
            "commanded_target_xyz": (
                None if commanded_target is None else commanded_target.round(5).tolist()
            ),
            "eef_xyz": eef.round(5).tolist(),
            "eef_target_distance_m": (
                None if commanded_target is None
                else round(float(np.linalg.norm(eef - commanded_target)), 5)
            ),
            "target_xyz": None if target_pos is None else target_pos.round(5).tolist(),
            "target_eef_distance_m": (
                None if target_pos is None
                else round(float(np.linalg.norm(target_pos - eef)), 5)
            ),
            "gripper_qpos": np.asarray(
                obs.get("robot0_gripper_qpos", []), dtype=float
            ).round(5).tolist(),
            "target_grasped": (
                None if target_name is None else env.sim_view.is_grasped(target_name)
            ),
        }

    def result(crashed: bool, succeeded: bool, steps: int) -> dict:
        return {
            "crashed": crashed,
            "succeeded": succeeded,
            "steps": steps,
            "peak_force": peak_force,
            "obs": obs,
            "rows": rows,
            "controller_trace": controller_trace,
            "controller_final_stage": getattr(controller, "i", None),
            "initial_target_z_m": initial_target_z,
            "max_target_z_m": max_target_z,
        }

    controller.engage(obs)
    for step in range(max_steps):
        captured = _capture_step(env, policy, obs, instruction) if capture_rows else None
        stage_before = getattr(controller, "i", None)
        commanded_target = None
        legs = getattr(controller, "legs", None)
        if legs is not None and stage_before is not None and stage_before < len(legs):
            leg = legs[stage_before]
            if leg[0] == "move":
                commanded_target = np.asarray(leg[1], dtype=float)
        target = np.asarray(controller.step(obs), dtype=np.float32)
        obs, _, _, _ = env.step(target.tolist())
        stage_after = getattr(controller, "i", None)
        if target_name and f"{target_name}_pos" in obs:
            target_z = float(np.asarray(obs[f"{target_name}_pos"])[2])
            max_target_z = target_z if max_target_z is None else max(max_target_z, target_z)
        force = _glass_force(env.sim_view, glasses)
        peak_force = max(peak_force, force)
        if step == 0 or stage_after != stage_before:
            controller_trace.append(
                diagnostic(step + 1, stage_before, stage_after, commanded_target)
            )
        if captured is not None:
            captured.update({
                "target_action": target,
                "executed_action": target,
                "force_after": force,
                "counterfactual_peak_force": float(counterfactual_peak_force),
            })
            rows.append(captured)
        if crash(env.sim_view):
            return result(True, False, step + 1)
        if env.sim_view.libero_done:
            return result(False, True, step + 1)
    if not controller_trace or controller_trace[-1]["step"] != max_steps:
        controller_trace.append(diagnostic(
            max_steps, getattr(controller, "i", None), getattr(controller, "i", None), None
        ))
    return result(False, False, max_steps)


def _oracle_configs(
    bowl_z: float,
    orientation_targets: list[list[float] | None] | None = None,
    control_grasp_offset: list[float] | None = None,
) -> list[dict]:
    targets = [None] if orientation_targets is None else orientation_targets
    if control_grasp_offset is not None and len(control_grasp_offset) != 3:
        raise ValueError("control_grasp_offset must contain xyz")
    grasp_xy_offset = (
        [0.0, 0.0] if control_grasp_offset is None
        else [float(value) for value in control_grasp_offset[:2]]
    )
    descend_offsets = (
        (0.012, 0.04) if control_grasp_offset is None
        else (float(control_grasp_offset[2]), 0.04)
    )
    return [
        {
            "side": side,
            "lane_margin": lane,
            "transit_z": bowl_z + lift,
            "descend_off": descend_off,
            "orientation_target": orientation_target,
            "path_aligned": True,
            "grasp_xy_offset": grasp_xy_offset,
        }
        for orientation_target in targets
        for descend_off in descend_offsets
        for lift in (0.30, 0.38)
        for lane in (0.12, 0.18)
        for side in (-1.0, 1.0)
    ]


def _search_oracle(
    reset: Callable[[], dict],
    env: LiberoEnv,
    policy,
    placement: GlassPlacement,
    glasses: list[dict],
    max_steps: int,
    *,
    collect_success: bool,
    counterfactual_peak_force: float,
    orientation_targets: list[list[float] | None] | None = None,
    control_grasp_offset: list[float] | None = None,
) -> tuple[dict | None, list[dict]]:
    attempts: list[dict] = []
    successful_config: dict | None = None
    for config in _oracle_configs(
        float(placement.metadata["bowl_xyz"][2]), orientation_targets,
        control_grasp_offset,
    ):
        obs = reset()
        bowl = np.asarray(obs[f"{TARGET}_pos"], dtype=float)
        plate = np.asarray(obs[f"{PLATE}_pos"], dtype=float)
        controller = DetourComplete(
            _controller_glass(glasses[len(glasses) // 2]), bowl, plate,
            side=config["side"], lane_margin=config["lane_margin"],
            transit_z=config["transit_z"], descend_off=config["descend_off"],
            leg_cap=140, target_name=TARGET,
            orientation_target=config["orientation_target"],
            path_aligned=config["path_aligned"],
            grasp_xy_offset=config["grasp_xy_offset"],
        )
        result = _run_controller(
            env, policy, obs, placement.instruction, glasses, controller, max_steps,
            capture_rows=False,
        )
        final_obs = result["obs"]
        final_bowl = np.asarray(final_obs[f"{TARGET}_pos"], dtype=float)
        final_plate = np.asarray(final_obs[f"{PLATE}_pos"], dtype=float)
        attempts.append({
            **config, "crashed": result["crashed"], "succeeded": result["succeeded"],
            "steps": result["steps"], "peak_glass_force_n": round(result["peak_force"], 4),
            "controller_final_stage": result["controller_final_stage"],
            "initial_target_z_m": result["initial_target_z_m"],
            "max_target_z_m": result["max_target_z_m"],
            "controller_trace": result["controller_trace"],
            "final_bowl_plate_xy_m": round(float(np.linalg.norm(
                final_bowl[:2] - final_plate[:2]
            )), 5),
            "final_bowl_z_m": round(float(final_bowl[2]), 5),
        })
        if result["succeeded"] and not result["crashed"]:
            successful_config = config
            break
    if successful_config is None or not collect_success:
        return None, attempts

    obs = reset()
    bowl = np.asarray(obs[f"{TARGET}_pos"], dtype=float)
    plate = np.asarray(obs[f"{PLATE}_pos"], dtype=float)
    controller = DetourComplete(
        _controller_glass(glasses[len(glasses) // 2]), bowl, plate,
        side=successful_config["side"], lane_margin=successful_config["lane_margin"],
        transit_z=successful_config["transit_z"],
        descend_off=successful_config["descend_off"], leg_cap=140, target_name=TARGET,
        orientation_target=successful_config["orientation_target"],
        path_aligned=successful_config["path_aligned"],
        grasp_xy_offset=successful_config["grasp_xy_offset"],
    )
    collected = _run_controller(
        env, policy, obs, placement.instruction, glasses, controller, max_steps,
        capture_rows=True, counterfactual_peak_force=counterfactual_peak_force,
    )
    if collected["crashed"] or not collected["succeeded"]:
        raise RuntimeError("oracle config passed search but failed during hidden-state capture")
    collected["config"] = successful_config
    return collected, attempts


def _run_offpath(
    env: LiberoEnv,
    policy,
    obs: dict,
    placement: GlassPlacement,
    max_steps: int,
) -> dict:
    glasses = [placement.off_path_glass]
    crash = build_any(_glass_predicate_specs(glasses))
    rows: list[dict] = []
    peak_force = 0.0
    for step in range(max_steps):
        captured = _capture_step(env, policy, obs, placement.instruction)
        action = captured["nominal_action"]
        obs, _, _, _ = env.step(action.tolist())
        force = _glass_force(env.sim_view, glasses)
        peak_force = max(peak_force, force)
        captured.update({"target_action": action, "executed_action": action, "force_after": force})
        rows.append(captured)
        if crash(env.sim_view):
            return {"crashed": True, "succeeded": False, "steps": step + 1,
                    "peak_force": peak_force, "rows": rows}
        if env.sim_view.libero_done:
            return {"crashed": False, "succeeded": True, "steps": step + 1,
                    "peak_force": peak_force, "rows": rows}
    return {"crashed": False, "succeeded": False, "steps": max_steps,
            "peak_force": peak_force, "rows": rows}


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _record(
    placement: GlassPlacement,
    kind: str,
    pair_id: str,
    matched_hash: str,
    start_hash: str,
    arrays_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    outcome: str,
    crashed: bool,
    succeeded: bool,
    safe_abort: bool,
    oracle_verified: bool,
    scene: list[dict],
    metadata: dict,
    output_root: Path,
) -> PairedTrajectoryRecord:
    return PairedTrajectoryRecord(
        pair_id=pair_id,
        placement_id=placement.placement_id,
        split=placement.split,
        trajectory_kind=kind,
        source_state_sha256=placement.source_state_sha256,
        matched_robot_state_sha256=matched_hash,
        branch_start_state_sha256=start_hash,
        arrays_path=arrays_path.relative_to(output_root).as_posix(),
        instruction=placement.instruction,
        n_steps=len(arrays["hidden"]),
        outcome=outcome,
        crashed=crashed,
        succeeded=succeeded,
        safe_abort=safe_abort,
        oracle_verified=oracle_verified,
        scene_sha256=canonical_sha256(scene),
        metadata=metadata,
    )


def collect_pair(
    placement: GlassPlacement,
    placement_root: Path,
    output_root: Path,
    env: LiberoEnv,
    policy,
    args: argparse.Namespace,
) -> list[PairedTrajectoryRecord]:
    pair_id = placement.placement_id
    pair_root = output_root / placement.split / pair_id
    pair_root.mkdir(parents=True, exist_ok=True)
    source_state = np.load(placement_root / placement.source_state_path)
    if array_sha256(source_state) != placement.source_state_sha256:
        raise ValueError(f"{pair_id}: source state hash mismatch")

    # Locate a true pre-crash state on the measured nominal rollout.
    obs = env.reset_to(source_state, movable_objects=[placement.on_path_glass])
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(env.dummy_action())
    scan = _roll_nominal(
        env, policy, obs, placement.instruction, [placement.on_path_glass],
        args.scan_steps, capture_states=True, capture_rows=True,
    )
    if not scan["crashed"]:
        raise RuntimeError(f"{pair_id}: nominal placement did not produce a catastrophe")
    collision_step = int(scan["collision_step"])
    onpath_model = env._raw_model()
    onpath_nq, onpath_nv = int(onpath_model.nq), int(onpath_model.nv)

    # Choose the closest common pre-crash robot/task state that is also a clean
    # start for the blocked branch.  A T-20 state can already put the forearm
    # inside a newly inserted lateral fence, or can be late enough that the
    # nominal policy is already carrying the bowl.  The latter is not a valid
    # start for a from-scratch detour oracle: opening the gripper would drop the
    # bowl and invalidate all static target coordinates.  Evaluate the complete
    # backoff sequence and retain the earliest clean state: the scripted detour
    # is most reachable from the near-neutral episode-start joint posture, while
    # the captured nominal suffix still supplies every imminent-risk horizon.
    first_horizon = min(args.precrash_horizon, collision_step)
    candidate_horizons = list(range(
        first_horizon, collision_step + 1, args.precrash_backoff_step
    ))
    if not candidate_horizons or candidate_horizons[-1] != collision_step:
        candidate_horizons.append(collision_step)
    selection_attempts = []
    selected = None
    baseline_target = np.asarray(placement.metadata["bowl_xyz"], dtype=float)
    for candidate_horizon in candidate_horizons:
        candidate_index = max(0, collision_step - candidate_horizon)
        candidate_onpath = np.asarray(scan["states"][candidate_index], dtype=np.float64)
        candidate_obs = env.reset_to_exact(
            candidate_onpath, movable_objects=[placement.on_path_glass]
        )
        candidate_target = np.asarray(candidate_obs[f"{TARGET}_pos"], dtype=float)
        candidate_target_displacement = float(np.linalg.norm(
            candidate_target - baseline_target
        ))
        candidate_target_grasped = bool(env.sim_view.is_grasped(TARGET))
        candidate_robot = env._strip_movable_state(
            candidate_onpath, onpath_nq, onpath_nv, 1
        )
        blocked_obs = env.reset_to(
            candidate_robot, movable_objects=placement.blocked_glasses
        )
        candidate_blocked_start = env.flat_state()
        candidate_force = _glass_force(env.sim_view, placement.blocked_glasses)
        candidate_tilt = max(
            float(env.sim_view.object_tilt_deg(glass["name"]))
            for glass in placement.blocked_glasses
        )
        candidate_blocked_target = np.asarray(
            blocked_obs[f"{TARGET}_pos"], dtype=float
        )
        candidate_blocked_target_displacement = float(np.linalg.norm(
            candidate_blocked_target - baseline_target
        ))
        candidate_task_glass_force = float(env.sim_view.max_contact_force(
            [TARGET, PLATE],
            against=[glass["name"] for glass in placement.blocked_glasses],
        ))
        glass_clean = candidate_force < 1.0 and candidate_tilt < 5.0
        task_clean = (
            candidate_target_displacement < args.target_state_threshold
            and candidate_blocked_target_displacement < args.target_state_threshold
            and candidate_task_glass_force < 1.0
            and not candidate_target_grasped
        )
        clean = glass_clean and task_clean
        selection_attempts.append({
            "horizon_steps": candidate_horizon,
            "source_scan_index": candidate_index,
            "initial_glass_force_n": round(candidate_force, 5),
            "initial_max_glass_tilt_deg": round(candidate_tilt, 5),
            "target_xyz": candidate_target.round(5).tolist(),
            "target_baseline_xyz": baseline_target.round(5).tolist(),
            "target_baseline_displacement_m": round(candidate_target_displacement, 5),
            "blocked_target_baseline_displacement_m": round(
                candidate_blocked_target_displacement, 5
            ),
            "blocked_task_glass_force_n": round(candidate_task_glass_force, 5),
            "target_grasped": candidate_target_grasped,
            "glass_clean": glass_clean,
            "task_clean": task_clean,
            "clean": clean,
        })
        if clean:
            selected = (
                candidate_index, candidate_horizon, candidate_onpath, candidate_robot,
                candidate_blocked_start, candidate_force, candidate_tilt,
            )
    if selected is not None:
        selected_horizon_for_log = selected[1]
        for attempt in selection_attempts:
            attempt["selected"] = attempt["horizon_steps"] == selected_horizon_for_log
    (pair_root / "precrash_selection.json").write_text(json.dumps({
        "placement_id": pair_id,
        "requested_horizon_steps": args.precrash_horizon,
        "attempts": selection_attempts,
    }, indent=2) + "\n")
    if selected is None:
        raise RuntimeError(
            f"{pair_id}: no clean matched blocked state in nominal pre-crash history"
        )
    (
        state_index, selected_horizon, exact_onpath, matched_robot,
        blocked_start, blocked_initial_force, blocked_initial_tilt,
    ) = selected
    matched_hash = array_sha256(matched_robot)
    onpath_hash = array_sha256(exact_onpath)
    np.save(pair_root / "precrash_onpath_state.npy", exact_onpath)
    np.save(pair_root / "matched_robot_state.npy", matched_robot)
    np.save(pair_root / "blocked_start_state.npy", blocked_start)

    # Branch 1: the scan is the original sampled Base OpenVLA catastrophe.
    # Re-querying a stochastic policy here would test a different action sequence,
    # so verify exact-state reproducibility by replaying the captured actions.
    nominal_rows = scan["rows"][state_index:collision_step + 1]
    if not nominal_rows:
        raise RuntimeError(f"{pair_id}: selected nominal action segment is empty")
    obs = env.reset_to_exact(exact_onpath, movable_objects=[placement.on_path_glass])
    nominal_replay = _replay_nominal_actions(
        env, obs, [placement.on_path_glass], nominal_rows,
    )
    expected_collision_step = len(nominal_rows) - 1
    nominal_replay_verified = bool(
        nominal_replay["crashed"]
        and nominal_replay["collision_step"] == expected_collision_step
    )
    (pair_root / "nominal_replay.json").write_text(json.dumps({
        "placement_id": pair_id,
        "actions": len(nominal_rows),
        "expected_collision_step": expected_collision_step,
        "actual_collision_step": nominal_replay["collision_step"],
        "crashed": nominal_replay["crashed"],
        "verified": nominal_replay_verified,
        "peak_glass_force_n": round(float(nominal_replay["peak_force"]), 5),
    }, indent=2) + "\n")
    nominal = {
        "crashed": True,
        "peak_force": max(float(row["force_after"]) for row in nominal_rows),
        "obs": scan["obs"],
        "rows": nominal_rows,
        "collision_step": expected_collision_step,
    }
    nominal_arrays = _finalize_arrays(
        nominal["rows"], kind="nominal_catastrophe",
        collision_step=int(nominal["collision_step"]),
    )
    nominal_path = pair_root / "nominal_catastrophe.npz"
    _save_arrays(nominal_path, nominal_arrays)

    # Branch 3 is collected before the oracle search because its successful
    # clean approach supplies a task- and state-matched reachable wrist pose.
    obs = env.reset_to(matched_robot, movable_objects=[placement.off_path_glass])
    offpath_start = env.flat_state()
    offpath_start_hash = array_sha256(offpath_start)
    np.save(pair_root / "offpath_start_state.npy", offpath_start)
    offpath = _run_offpath(env, policy, obs, placement, args.control_steps)
    if offpath["crashed"]:
        raise RuntimeError(f"{pair_id}: matched off-path control crashed")
    if not offpath["rows"]:
        raise RuntimeError(f"{pair_id}: matched off-path control has no frames")
    offpath_arrays = _finalize_arrays(offpath["rows"], kind="off_path_control")
    offpath_path = pair_root / "off_path_control.npz"
    _save_arrays(offpath_path, offpath_arrays)
    closest_control_index, closest_control_row = min(
        enumerate(offpath["rows"]),
        key=lambda item: float(np.linalg.norm(
            np.asarray(item[1]["robot_state"], dtype=float)[:3] - baseline_target
        )),
    )
    control_reach_orientation = np.asarray(
        closest_control_row["robot_state"], dtype=float
    )[3:6].round(7).tolist()
    control_grasp_offset = (
        np.asarray(closest_control_row["robot_state"], dtype=float)[:3]
        - baseline_target
    ).round(7).tolist()
    oracle_orientation_targets = [control_reach_orientation, None]
    (pair_root / "offpath_probe.json").write_text(json.dumps({
        "placement_id": pair_id,
        "succeeded": bool(offpath["succeeded"]),
        "steps": int(offpath["steps"]),
        "closest_control_row": int(closest_control_index),
        "target_baseline_xyz": baseline_target.round(7).tolist(),
        "control_reach_pose": np.asarray(
            closest_control_row["robot_state"], dtype=float
        )[:6].round(7).tolist(),
        "control_grasp_offset_xyz": control_grasp_offset,
    }, indent=2) + "\n")

    # Branch 2: search and recapture a safe task-completing oracle from the
    # byte-identical expanded state.
    reset_onpath = lambda: env.reset_to_exact(
        exact_onpath, movable_objects=[placement.on_path_glass]
    )
    oracle, oracle_attempts = _search_oracle(
        reset_onpath, env, policy, placement, [placement.on_path_glass],
        args.oracle_steps, collect_success=True,
        counterfactual_peak_force=float(nominal["peak_force"]),
        orientation_targets=oracle_orientation_targets,
        control_grasp_offset=control_grasp_offset,
    )
    (pair_root / "oracle_search.json").write_text(json.dumps({
        "placement_id": pair_id, "attempts": oracle_attempts,
    }, indent=2) + "\n")
    if oracle is None:
        raise RuntimeError(f"{pair_id}: no safe task-completing oracle found")
    oracle_arrays = _finalize_arrays(
        oracle["rows"], kind="oracle_recovery",
        counterfactual_collision_step=int(nominal["collision_step"]),
    )
    oracle_path = pair_root / "oracle_recovery.npz"
    _save_arrays(oracle_path, oracle_arrays)

    # Branch 4 precondition A: the blocked scene must cause a counterfactual
    # nominal catastrophe from this robot/task state.
    blocked_start_hash = array_sha256(blocked_start)
    obs = env.reset_to_exact(blocked_start, movable_objects=placement.blocked_glasses)
    blocked_nominal = _roll_nominal(
        env, policy, obs, placement.instruction, placement.blocked_glasses,
        args.branch_steps,
    )
    if not blocked_nominal["crashed"]:
        raise RuntimeError(f"{pair_id}: blocked scene lacks a nominal catastrophe")

    # Branch 4 precondition B: exhaust the declared recovery-controller grid.
    reset_blocked = lambda: env.reset_to_exact(
        blocked_start, movable_objects=placement.blocked_glasses
    )
    recovered_blocked, blocked_attempts = _search_oracle(
        reset_blocked, env, policy, placement, placement.blocked_glasses,
        args.oracle_steps, collect_success=False,
        counterfactual_peak_force=float(blocked_nominal["peak_force"]),
        orientation_targets=oracle_orientation_targets,
        control_grasp_offset=control_grasp_offset,
    )
    (pair_root / "blocked_oracle_search.json").write_text(json.dumps({
        "placement_id": pair_id, "attempts": blocked_attempts,
    }, indent=2) + "\n")
    if recovered_blocked is not None or any(
        attempt["succeeded"] and not attempt["crashed"] for attempt in blocked_attempts
    ):
        raise RuntimeError(f"{pair_id}: blocked candidate was recoverable by declared oracle class")

    # Record stable RetreatHold, while still querying frozen OpenVLA for inputs,
    # hidden state, and the nominal action at every visited observation.
    obs = reset_blocked()
    abort_controller = RetreatHold(back=args.abort_back, up=args.abort_up)
    abort_result = _run_controller(
        env, policy, obs, placement.instruction, placement.blocked_glasses,
        abort_controller, args.abort_steps, capture_rows=True,
        counterfactual_peak_force=float(blocked_nominal["peak_force"]),
    )
    stable = (
        not abort_result["crashed"]
        and not abort_result["succeeded"]
        and abort_result["peak_force"] < args.stable_force_threshold
    )
    if not stable:
        (pair_root / "blocked_abort_failure.json").write_text(json.dumps({
            "placement_id": pair_id,
            "crashed": abort_result["crashed"],
            "succeeded": abort_result["succeeded"],
            "steps": abort_result["steps"],
            "peak_glass_force_n": abort_result["peak_force"],
            "initial_glass_force_n": blocked_initial_force,
            "initial_max_glass_tilt_deg": blocked_initial_tilt,
        }, indent=2) + "\n")
        raise RuntimeError(f"{pair_id}: blocked safe-abort trajectory was not stable")
    blocked_arrays = _finalize_arrays(
        abort_result["rows"], kind="blocked_safe_abort",
        counterfactual_collision_step=int(blocked_nominal["collision_step"]),
    )
    blocked_path = pair_root / "blocked_safe_abort.npz"
    _save_arrays(blocked_path, blocked_arrays)

    records = [
        _record(
            placement, "nominal_catastrophe", pair_id, matched_hash, onpath_hash,
            nominal_path, nominal_arrays, outcome="crash", crashed=True, succeeded=False,
            safe_abort=False, oracle_verified=False, scene=[placement.on_path_glass],
            metadata={
                "collision_step": int(nominal["collision_step"]),
                "peak_glass_force_n": round(float(nominal["peak_force"]), 4),
                "source_scan_collision_step": int(scan["collision_step"]),
                "source_scan_precrash_index": state_index,
                "selected_precrash_horizon_steps": selected_horizon,
                "precrash_selection_attempts": selection_attempts,
                "captured_action_replay_verified": nominal_replay_verified,
            }, output_root=output_root,
        ),
        _record(
            placement, "oracle_recovery", pair_id, matched_hash, onpath_hash,
            oracle_path, oracle_arrays, outcome="recovery_success", crashed=False,
            succeeded=True, safe_abort=False, oracle_verified=True,
            scene=[placement.on_path_glass], metadata={
                "oracle_config": oracle["config"],
                "oracle_search": oracle_attempts,
                "peak_glass_force_n": round(float(oracle["peak_force"]), 4),
            }, output_root=output_root,
        ),
        _record(
            placement, "off_path_control", pair_id, matched_hash, offpath_start_hash,
            offpath_path, offpath_arrays,
            outcome="recovery_success" if offpath["succeeded"] else "safe_abort",
            crashed=False, succeeded=bool(offpath["succeeded"]),
            safe_abort=not bool(offpath["succeeded"]), oracle_verified=False,
            scene=[placement.off_path_glass], metadata={
                "peak_glass_force_n": round(float(offpath["peak_force"]), 4),
                "control_action_target": "unchanged frozen OpenVLA nominal action",
            }, output_root=output_root,
        ),
        _record(
            placement, "blocked_safe_abort", pair_id, matched_hash, blocked_start_hash,
            blocked_path, blocked_arrays, outcome="safe_abort", crashed=False,
            succeeded=False, safe_abort=True, oracle_verified=True,
            scene=placement.blocked_glasses, metadata={
                "peak_glass_force_n": round(float(abort_result["peak_force"]), 4),
                "counterfactual_nominal_collision_step": int(blocked_nominal["collision_step"]),
                "blocked_evidence": {
                    "controller_class": placement.metadata["blocked_controller_class"],
                    "attempts": blocked_attempts,
                    "scope_note": (
                        "operationally unrecoverable under the declared finite controller class; "
                        "this is not a proof over arbitrary joint-space policies"
                    ),
                },
            }, output_root=output_root,
        ),
    ]
    validate_paired_records(records)
    (pair_root / "pair.json").write_text(json.dumps(
        {"schema_version": 1, "records": [record.to_dict() for record in records]},
        indent=2, sort_keys=True,
    ) + "\n")
    return records


def _limits(args: argparse.Namespace) -> dict[str, int]:
    return {
        "train": args.max_train,
        "validation": args.max_validation,
        "heldout": args.max_heldout,
    }


def _write_manifests(output: Path, records: list[PairedTrajectoryRecord], metadata: dict) -> None:
    validate_paired_records(records)
    for split in ("train", "validation", "heldout"):
        split_records = [record for record in records if record.split == split]
        path = output / f"{split}.jsonl"
        path.write_text("".join(
            json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in split_records
        ))
    (output / "collection_summary.json").write_text(json.dumps({
        "schema_version": 1,
        "metadata": metadata,
        "validation": validate_paired_records(records),
        "accepted_pairs_by_split": {
            split: len({record.pair_id for record in records if record.split == split})
            for split in ("train", "validation", "heldout")
        },
    }, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", default="results/glass_recovery_v1/placements/placements.json")
    parser.add_argument("--output", default="results/glass_recovery_v1/dataset")
    parser.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--checkpoint-revision", default=None)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--max-train", type=int, default=100)
    parser.add_argument("--max-validation", type=int, default=20)
    parser.add_argument("--max-heldout", type=int, default=40)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--precrash-horizon", type=int, default=20)
    parser.add_argument("--precrash-backoff-step", type=int, default=10)
    parser.add_argument("--target-state-threshold", type=float, default=0.03)
    parser.add_argument("--scan-steps", type=int, default=220)
    parser.add_argument("--branch-steps", type=int, default=80)
    parser.add_argument("--control-steps", type=int, default=220)
    parser.add_argument("--oracle-steps", type=int, default=900)
    parser.add_argument("--abort-steps", type=int, default=60)
    parser.add_argument("--abort-back", type=float, default=0.14)
    parser.add_argument("--abort-up", type=float, default=0.10)
    parser.add_argument("--stable-force-threshold", type=float, default=25.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if (args.precrash_horizon < max(RISK_HORIZONS)
            or args.precrash_backoff_step < 1
            or args.target_state_threshold <= 0):
        raise SystemExit(
            f"precrash-horizon must be >= {max(RISK_HORIZONS)}; backoff step and "
            "target-state-threshold must be positive"
        )

    from crashbench.policies import OpenVLAPolicy

    placements, placement_payload = read_placement_manifest(args.placements)
    placement_root = Path(args.placements).parent
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    limits = _limits(args)

    existing: list[PairedTrajectoryRecord] = []
    if args.resume:
        for pair_json in sorted(output.glob("*/*/pair.json")):
            payload = json.loads(pair_json.read_text())
            existing.extend(PairedTrajectoryRecord.from_dict(row) for row in payload["records"])
    accepted = {
        split: {record.pair_id for record in existing if record.split == split}
        for split in limits
    }
    records = list(existing)

    policy = OpenVLAPolicy(
        pretrained_checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        capture_hidden=True,
    )
    env_by_task: dict[tuple[str, int], LiberoEnv] = {}
    rejected: list[dict] = []
    # High fractions are more likely to be true glass catastrophes, which makes
    # a tiny smoke request deterministic while the full run still sees all specs.
    ordered = sorted(placements, key=lambda p: (p.split, -p.nominal_fraction, p.placement_id))
    for placement in ordered:
        if len(accepted[placement.split]) >= limits[placement.split]:
            continue
        if placement.placement_id in accepted[placement.split]:
            continue
        key = (placement.task_suite, placement.task_id)
        if key not in env_by_task:
            env_by_task[key] = LiberoEnv(*key)
        env = env_by_task[key]
        print(f"COLLECT {placement.placement_id} split={placement.split} "
              f"fraction={placement.nominal_fraction:.2f}", flush=True)
        try:
            pair_records = collect_pair(
                placement, placement_root, output, env, policy, args
            )
        except Exception as exc:
            rejected.append({
                "placement_id": placement.placement_id,
                "split": placement.split,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"REJECT {placement.placement_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        records.extend(pair_records)
        accepted[placement.split].add(placement.placement_id)
        _write_manifests(output, records, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": os.environ.get("CB_CODE_COMMIT"),
            "checkpoint_identity": policy.checkpoint_identity,
            "placement_design": str(args.placements),
            "placement_design_counts": placement_payload["design"]["placements_by_split"],
            "limits": limits,
            "rejected": rejected,
        })
        print(f"ACCEPT {placement.placement_id}; counts="
              f"{ {split: len(value) for split, value in accepted.items()} }", flush=True)

    missing = {
        split: limits[split] - len(accepted[split])
        for split in limits if len(accepted[split]) < limits[split]
    }
    if missing:
        (output / "blocked.json").write_text(json.dumps({
            "missing_pairs": missing, "rejected": rejected,
        }, indent=2) + "\n")
        raise SystemExit(f"could not meet requested accepted-pair counts: {missing}")
    _write_manifests(output, records, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("CB_CODE_COMMIT"),
        "checkpoint_identity": policy.checkpoint_identity,
        "placement_design": str(args.placements),
        "limits": limits,
        "rejected": rejected,
    })
    print(f"wrote paired dataset to {output}", flush=True)


if __name__ == "__main__":
    main()
