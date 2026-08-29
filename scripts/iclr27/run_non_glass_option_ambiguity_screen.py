#!/usr/bin/env python3
"""Freeze and execute one bounded exact-state unstable-placement screen.

The real path first evaluates the complete prospective nominal-source list,
freezes source hashes, geometry, parameters, horizons, predicates, controller
settings, and the 108 linked-decision grid, and only then opens non-Base
branches.  ``no_hazard`` is physically executed once per source/horizon and
linked to all three matched parameter blocks without repeated evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES
from crashbench.provenance import (
    repository_provenance,
    require_checkpoint_revision,
    runtime_provenance,
)
from crashbench.unstable_placement import (
    CONDITIONS,
    FAMILY_ID,
    OPTIONS,
    SCHEMA_VERSION,
    SEVERITY_IDS,
    HeldObjectPlacementController,
    StabilityOutcomeTracker,
    array_sha256,
    build_attempted_grid,
    build_support_patch,
    canonical_sha256,
    choose_auxiliary_locations,
    derive_geometry,
    evaluate_base_repeat_audit,
    patch_surface_upper_z,
    patch_xy_half_extents,
    validate_screen_config,
)
from scripts.collect_glass_recovery_pairs import (
    _branch_start_hashes,
    _controller_state_sha256,
    _continuation_state_sha256,
    _observation_sha256,
    _require_branch_start_hashes,
)
from scripts.iclr27.analyze_non_glass_option_ambiguity_screen import analyze_screen


DEFAULT_CONFIG = ROOT / "configs/iclr27/non_glass_unstable_placement_screen_v1.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _path_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _touch_exclusive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8"):
        pass


def create_immutable_result_root(parent: str | Path, protocol_sha256: str, stamp: str) -> Path:
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / f"non_glass_option_ambiguity_screen_{protocol_sha256[:12]}_{stamp}"
    root.mkdir(mode=0o755)
    return root


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_rng_state() -> dict[str, np.ndarray]:
    state = {
        "python": np.frombuffer(pickle.dumps(random.getstate(), protocol=5), dtype=np.uint8).copy(),
        "numpy": np.frombuffer(pickle.dumps(np.random.get_state(), protocol=5), dtype=np.uint8).copy(),
    }
    try:
        import torch
    except ImportError:
        return state
    state["torch_cpu"] = torch.random.get_rng_state().cpu().numpy().copy()
    if torch.cuda.is_available():
        for index, value in enumerate(torch.cuda.get_rng_state_all()):
            state[f"torch_cuda_{index}"] = value.cpu().numpy().copy()
    return state


def _rng_state_sha256(state: Mapping[str, np.ndarray]) -> str:
    return canonical_sha256({key: array_sha256(value) for key, value in sorted(state.items())})


def _restore_rng_state(state: Mapping[str, np.ndarray]) -> None:
    random.setstate(pickle.loads(np.asarray(state["python"], dtype=np.uint8).tobytes()))
    np.random.set_state(pickle.loads(np.asarray(state["numpy"], dtype=np.uint8).tobytes()))
    try:
        import torch
    except ImportError:
        return
    if "torch_cpu" in state:
        torch.random.set_rng_state(torch.as_tensor(np.asarray(state["torch_cpu"]), dtype=torch.uint8))
    cuda_keys = sorted(
        (key for key in state if key.startswith("torch_cuda_")),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    if cuda_keys and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([
            torch.as_tensor(np.asarray(state[key]), dtype=torch.uint8) for key in cuda_keys
        ])


def _policy_continuation_sha256(
    *, rng_sha256: str, checkpoint_revision: str, instruction: str,
) -> str:
    return canonical_sha256({
        "policy": "frozen OpenVLA single-frame continuation",
        "temporal_state": "stateless; policy.reset() before every exact branch",
        "rng_state_sha256": rng_sha256,
        "checkpoint_revision": checkpoint_revision,
        "instruction": instruction,
    })


def _complete_branch_hash(
    environment_hashes: Mapping[str, str], *, rng_sha256: str,
    policy_continuation_sha256: str,
) -> str:
    return canonical_sha256({
        **dict(environment_hashes),
        "rng_state_sha256": rng_sha256,
        "policy_continuation_state_sha256": policy_continuation_sha256,
    })


def _body_descendants(model, root_id: int) -> set[int]:
    selected = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if int(model.body_parentid[body_id]) in selected and body_id not in selected:
                selected.add(body_id)
                changed = True
    return selected


def _world_geom_bounds(model, data, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(model.geom_aabb[geom_id], dtype=float)
    center_local, half_local = local[:3], local[3:]
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    center = np.asarray(data.geom_xpos[geom_id], dtype=float) + rotation @ center_local
    half = np.abs(rotation) @ half_local
    if not np.any(half > 0):
        half = np.full(3, float(model.geom_rbound[geom_id]))
    return center - half, center + half


def _body_bounds(env: LiberoEnv, name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import mujoco

    model, data = env.sim_view._live_mj()
    root_id = env.sim_view._body_id(model, name)
    bodies = _body_descendants(model, root_id)
    bounds = []
    names = []
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom_id]) not in bodies:
            continue
        bounds.append(_world_geom_bounds(model, data, geom_id))
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}")
    if not bounds:
        raise ValueError(f"no MuJoCo geoms found for body {name!r}")
    return (
        np.min(np.stack([row[0] for row in bounds]), axis=0),
        np.max(np.stack([row[1] for row in bounds]), axis=0),
        names,
    )


def _table_geometry(env: LiberoEnv, reference_z: float) -> dict[str, Any]:
    import mujoco

    model, data = env.sim_view._live_mj()
    candidates = []
    for geom_id in range(int(model.ngeom)):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        lo, hi = _world_geom_bounds(model, data, geom_id)
        extent = hi - lo
        area = float(extent[0] * extent[1])
        named = "table" in geom_name.lower() or "table" in body_name.lower()
        if (named or (extent[0] > 0.35 and extent[1] > 0.35)) and hi[2] <= reference_z + 0.08:
            candidates.append((int(named), area, float(hi[2]), geom_name, body_name, lo, hi))
    if not candidates:
        raise ValueError("could not identify the LIBERO table support geom")
    candidates.sort(key=lambda row: (-row[0], -row[1], abs(reference_z - row[2]), row[3]))
    _, _, top, geom_name, body_name, lo, hi = candidates[0]
    return {
        "table_top_z_m": top,
        "table_bounds_xy": [lo[:2].tolist(), hi[:2].tolist()],
        "table_geom_name": geom_name,
        "table_body_name": body_name,
    }


def _measure_geometry(env: LiberoEnv, obs: Mapping[str, Any], target: str, support: str) -> dict[str, Any]:
    bowl_lo, bowl_hi, bowl_geoms = _body_bounds(env, target)
    plate_lo, plate_hi, plate_geoms = _body_bounds(env, support)
    bowl_extent = bowl_hi - bowl_lo
    plate_extent = plate_hi - plate_lo
    bowl_radius = 0.5 * float(min(bowl_extent[0], bowl_extent[1]))
    bowl_half_height = 0.5 * float(bowl_extent[2])
    plate_radius = 0.5 * float(min(plate_extent[0], plate_extent[1]))
    table = _table_geometry(env, reference_z=float(plate_hi[2]))
    return {
        "bowl_footprint_radius_m": bowl_radius,
        "bowl_half_height_m": bowl_half_height,
        "plate_support_radius_m": plate_radius,
        "plate_top_z_m": float(plate_hi[2]),
        "plate_center_xyz": np.asarray(obs[f"{support}_pos"], dtype=float).tolist(),
        "bowl_initial_center_xyz": np.asarray(obs[f"{target}_pos"], dtype=float).tolist(),
        "bowl_bounds_world": [bowl_lo.tolist(), bowl_hi.tolist()],
        "plate_bounds_world": [plate_lo.tolist(), plate_hi.tolist()],
        "bowl_geom_names": bowl_geoms,
        "plate_geom_names": plate_geoms,
        **table,
    }


def _camera_contract(env: LiberoEnv, obs: Mapping[str, Any], resize_size: int) -> dict[str, Any]:
    import mujoco

    model, _data = env.sim_view._live_mj()
    cameras = []
    for camera_id in range(int(model.ncam)):
        cameras.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id) or f"camera_{camera_id}",
            "pos": np.asarray(model.cam_pos[camera_id], dtype=float).round(10).tolist(),
            "quat": np.asarray(model.cam_quat[camera_id], dtype=float).round(10).tolist(),
            "fovy": float(model.cam_fovy[camera_id]),
        })
    image_shapes = {
        key: list(np.asarray(value).shape)
        for key, value in sorted(obs.items()) if key.endswith("_image")
    }
    payload = {
        "cameras": cameras,
        "raw_image_shapes": image_shapes,
        "policy_resize_size": int(resize_size),
        "center_crop": True,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _task_object_positions(obs: Mapping[str, Any]) -> dict[str, list[float]]:
    rows = {}
    for key, value in obs.items():
        array = np.asarray(value)
        if (
            key.endswith("_pos") and "_to_" not in key and not key.startswith("robot")
            and array.shape == (3,)
        ):
            rows[key] = array.astype(float).tolist()
    return rows


def _target_hazard_contact(env: LiberoEnv, target: str, hazard_name: str) -> bool:
    """Return whether the target-object subtree currently contacts the patch."""

    import mujoco

    model, data = env.sim_view._live_mj()
    target_ids = _body_descendants(model, env.sim_view._body_id(model, target))
    hazard_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hazard_name)
    if hazard_id < 0:
        raise KeyError(f"missing injected hazard body {hazard_name!r}")
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        left = int(model.geom_bodyid[contact.geom1])
        right = int(model.geom_bodyid[contact.geom2])
        if (left in target_ids and right == hazard_id) or (
            right in target_ids and left == hazard_id
        ):
            return True
    return False


def _held(env: LiberoEnv, obs: Mapping[str, Any], target: str) -> bool:
    if env.sim_view.is_grasped(target):
        return True
    object_pos = np.asarray(obs[f"{target}_pos"], dtype=float)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
    gripper = np.asarray(obs.get("robot0_gripper_qpos", [1.0, 1.0]), dtype=float)
    return bool(np.linalg.norm(object_pos - eef) < 0.075 and np.sum(np.abs(gripper)) < 0.065)


def _robot_swept_bounds(env: LiberoEnv, target_xyz: Sequence[float], prior: list[np.ndarray]) -> None:
    for row in env.sim_view.robot_geom_aabbs(list(ROBOT_CONTACT_BODIES)):
        prior.append(np.stack((np.asarray(row["lo"], dtype=float)[:2], np.asarray(row["hi"], dtype=float)[:2])))
    target = np.asarray(target_xyz, dtype=float)[:2]
    prior.append(np.stack((target, target)))


def _scan_nominal(
    env: LiberoEnv,
    policy,
    source_state: np.ndarray,
    *,
    instruction: str,
    target: str,
    support: str,
    seed: int,
    settle_steps: int,
    max_steps: int,
    release_confirm_steps: int,
    stability_dwell_steps: int,
    obstacle: Mapping[str, Any] | None,
    geometry: Mapping[str, Any] | None,
    capture_exact: bool,
) -> dict[str, Any]:
    _seed_everything(seed)
    env.seed(seed)
    policy.reset()
    obs = env.reset_to(source_state, obstacles=None if obstacle is None else [dict(obstacle)])
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(env.dummy_action())
    initial_obs = {key: np.asarray(value).copy() for key, value in obs.items()}
    measured = _measure_geometry(env, obs, target, support)
    camera = _camera_contract(env, obs, policy.resize_size)
    model_xml = env.model_xml()
    initial_robot_force = 0.0 if obstacle is None else float(env.sim_view.max_contact_force(
        list(ROBOT_CONTACT_BODIES), against=[str(obstacle["name"])],
    ))
    initial_target_contact = False if obstacle is None else _target_hazard_contact(
        env, target, str(obstacle["name"]),
    )
    states: list[np.ndarray] = []
    controller_states: list[dict[str, np.ndarray]] = []
    observations: list[dict[str, np.ndarray]] = []
    rng_states: list[dict[str, np.ndarray]] = []
    nominal_actions: list[np.ndarray] = []
    grasped_before: list[bool] = []
    target_xyz_before: list[np.ndarray] = []
    target_eef_relative_before: list[np.ndarray] = []
    target_tilt_before: list[float] = []
    swept: list[np.ndarray] = []
    task_success_seen = False
    ever_held = False
    ungrasped_run = 0
    release_action_index = None
    release_position = None
    dwell_positions = []
    dwell_tilts = []
    frames = [env.render(obs, 256)]
    termination = "max_steps"

    for step in range(max_steps):
        grasped = _held(env, obs, target)
        ever_held = ever_held or grasped
        target_xyz = np.asarray(obs[f"{target}_pos"], dtype=float).copy()
        eef_xyz = np.asarray(obs["robot0_eef_pos"], dtype=float).copy()
        target_xyz_before.append(target_xyz)
        target_eef_relative_before.append(target_xyz - eef_xyz)
        target_tilt_before.append(float(env.sim_view.object_tilt_deg(target)))
        _robot_swept_bounds(env, obs[f"{target}_pos"], swept)
        if capture_exact:
            states.append(env.flat_state())
            controller_states.append(env.controller_state())
            observations.append({key: np.asarray(value).copy() for key, value in obs.items()})
            rng_states.append(_capture_rng_state())
        grasped_before.append(grasped)
        policy_obs = env.policy_observation(obs, policy.resize_size)
        action = np.asarray(policy.act(policy_obs, instruction), dtype=np.float32)
        nominal_actions.append(action)
        obs, _, done, _ = env.step(action.tolist())
        task_success_seen = task_success_seen or bool(done)
        grasped_after = _held(env, obs, target)
        if ever_held:
            ungrasped_run = 0 if grasped_after else ungrasped_run + 1
            if release_action_index is None and ungrasped_run >= release_confirm_steps:
                release_action_index = step - release_confirm_steps + 1
                release_position = np.asarray(obs[f"{target}_pos"], dtype=float).copy()
                termination = "release_observed"
                break
        if env.episode_terminated():
            termination = "environment_horizon_before_release"
            break
        if step % 8 == 0:
            frames.append(env.render(obs, 256))

    # The fixed nominal-success gate adds an open-gripper validation dwell after
    # the semantic release event; these are not policy/anchor rows.
    if release_action_index is not None:
        for _ in range(stability_dwell_steps):
            obs, _, done, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            task_success_seen = task_success_seen or bool(done)
            dwell_positions.append(np.asarray(obs[f"{target}_pos"], dtype=float).copy())
            dwell_tilts.append(float(env.sim_view.object_tilt_deg(target)))
            if env.episode_terminated():
                break
    frames.append(env.render(obs, 256))

    if swept:
        swept_lo = np.min(np.stack([row[0] for row in swept]), axis=0)
        swept_hi = np.max(np.stack([row[1] for row in swept]), axis=0)
    else:
        swept_lo = swept_hi = np.asarray(obs[f"{target}_pos"], dtype=float)[:2]
    stable_nominal = False
    nominal_failure = None
    if release_action_index is None:
        nominal_failure = "no_held_to_released_transition"
    elif not task_success_seen:
        nominal_failure = "no_nominal_task_success"
    elif geometry is not None and dwell_positions:
        plate_xy = np.asarray(measured["plate_center_xyz"], dtype=float)[:2]
        final = dwell_positions[-1]
        strict_radius = float(geometry["strict_success_com_radius_m"])
        stable_nominal = bool(
            len(dwell_positions) >= stability_dwell_steps
            and np.all(np.asarray(dwell_tilts) <= float(geometry.get("stable_tilt_deg", 18.0)))
            and np.all(np.asarray(dwell_positions)[:, 2] >= float(geometry["fall_com_height_threshold_m"]))
            and float(np.linalg.norm(final[:2] - plate_xy)) <= strict_radius
        )
        if not stable_nominal:
            nominal_failure = "nominal_success_failed_stability_contract"
    elif geometry is None:
        # First pass geometry is resolved immediately after this scan; basic
        # success/release/dwell validity is sufficient for its raw record.
        stable_nominal = bool(len(dwell_positions) >= stability_dwell_steps)
        if not stable_nominal:
            nominal_failure = "incomplete_nominal_stability_dwell"

    return {
        "initial_observation": initial_obs,
        "measured_geometry": measured,
        "camera_contract": camera,
        "model_xml": model_xml,
        "model_xml_sha256": hashlib.sha256(model_xml.encode("utf-8")).hexdigest(),
        "initial_robot_hazard_force_n": initial_robot_force,
        "initial_target_hazard_contact": initial_target_contact,
        "states": states,
        "controller_states": controller_states,
        "observations": observations,
        "rng_states": rng_states,
        "nominal_actions": nominal_actions,
        "grasped_before": grasped_before,
        "target_xyz_before": target_xyz_before,
        "target_eef_relative_before": target_eef_relative_before,
        "target_tilt_before": target_tilt_before,
        "release_action_index": release_action_index,
        "release_position": None if release_position is None else release_position.tolist(),
        "task_success_seen": task_success_seen,
        "stable_nominal_success": stable_nominal,
        "nominal_failure": nominal_failure,
        "termination": termination,
        "swept_bounds_xy": [swept_lo.tolist(), swept_hi.tolist()],
        "task_object_positions": _task_object_positions(initial_obs),
        "frames": frames,
    }


def _geometry_signature(measured: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(measured[key]) for key in (
            "bowl_footprint_radius_m", "bowl_half_height_m",
            "plate_support_radius_m", "plate_top_z_m", "table_top_z_m",
        )
    }


def _source_scan_record(
    candidate: Mapping[str, Any], source_state: np.ndarray, scan: Mapping[str, Any],
    horizons: Sequence[int],
) -> dict[str, Any]:
    release = scan["release_action_index"]
    anchor_indices = {
        str(horizon): None if release is None else int(release) - int(horizon) + 1
        for horizon in horizons
    }
    failure = scan["nominal_failure"]
    if failure is None and any(value is None or value < 0 for value in anchor_indices.values()):
        failure = "nominal_release_before_predeclared_anchor"
    return {
        **dict(candidate),
        "source_state_sha256": array_sha256(source_state),
        "nominal_release_action_index": release,
        "nominal_anchor_indices": anchor_indices,
        "nominal_task_success": bool(scan["task_success_seen"]),
        "nominal_stable_success": bool(scan["stable_nominal_success"]),
        "nominal_steps": len(scan["nominal_actions"]),
        "nominal_termination": scan["termination"],
        "mechanically_valid": failure is None,
        "rejection_reason": failure,
        "geometry_signature": _geometry_signature(scan["measured_geometry"]),
        "camera_contract_sha256": scan["camera_contract"]["sha256"],
        "swept_bounds_xy": scan["swept_bounds_xy"],
    }


def _save_gif(path: Path, frames: Sequence[np.ndarray], *, stride: int = 1) -> str | None:
    try:
        import imageio.v2 as imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        selected = [np.asarray(frame).astype(np.uint8) for frame in frames[::max(1, stride)]]
        if frames and (not selected or not np.array_equal(selected[-1], frames[-1])):
            selected.append(np.asarray(frames[-1]).astype(np.uint8))
        imageio.mimsave(path, selected, duration=0.12)
        return path.name
    except Exception as exc:
        return f"media_error:{type(exc).__name__}:{exc}"


def freeze_protocol(
    *, output_parent: str | Path, config_path: str | Path,
    checkpoint_revision: str, mock: bool = False,
) -> Path:
    if mock:
        return _write_mock_run(output_parent=output_parent, config_path=config_path)
    config = json.loads(Path(config_path).read_text())
    accounting = validate_screen_config(config)
    revision = require_checkpoint_revision(checkpoint_revision)
    repo = repository_provenance(ROOT, require_clean=True)
    declared = os.environ.get("CB_CODE_COMMIT")
    if declared is not None and declared != repo["git_commit"]:
        raise RuntimeError("CB_CODE_COMMIT does not match the checked-out source")

    output_parent = Path(output_parent).resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = output_parent / f".non_glass_screen_freeze_{os.getpid()}_{_path_stamp()}"
    staging.mkdir()
    (staging / "logs").mkdir()
    (staging / "states").mkdir()
    (staging / "media").mkdir()
    try:
        from crashbench.policies import OpenVLAPolicy

        task = config["task"]
        env = LiberoEnv(task["suite"], int(task["task_id"]), seed=int(config["execution"]["rollout_seed"]))
        if env.task_description != task["instruction"]:
            raise RuntimeError(
                f"task instruction drift: {env.task_description!r} != {task['instruction']!r}"
            )
        policy = OpenVLAPolicy(
            pretrained_checkpoint=config["execution"]["checkpoint"],
            checkpoint_revision=revision,
            unnorm_key=config["execution"]["unnorm_key"],
            center_crop=True,
            capture_hidden=False,
        )
        default_states = np.asarray(env.default_init_states())
        candidate_rows = []
        passing_payloads = []
        raw_geometry = None
        geometry = None
        for order, candidate in enumerate(task["source_candidates"]):
            index = int(candidate["state_index"])
            if not 0 <= index < len(default_states):
                row = {**candidate, "candidate_order": order, "mechanically_valid": False, "rejection_reason": "state_index_out_of_range"}
                candidate_rows.append(row)
                continue
            state = np.asarray(default_states[index], dtype=np.float64)
            try:
                scan = _scan_nominal(
                    env, policy, state,
                    instruction=task["instruction"], target=task["target_object"],
                    support=task["final_support_object"], seed=int(candidate["reset_seed"]),
                    settle_steps=int(task["settle_steps"]), max_steps=int(task["nominal_max_steps"]),
                    release_confirm_steps=int(config["predicates"]["release_confirm_steps"]),
                    stability_dwell_steps=int(config["predicates"]["stability_dwell_steps"]),
                    obstacle=None, geometry=geometry, capture_exact=False,
                )
                if raw_geometry is None:
                    raw_geometry = scan["measured_geometry"]
                    geometry = derive_geometry(raw_geometry, config)
                    geometry["stable_tilt_deg"] = float(config["predicates"]["stable_tilt_deg"])
                    # Re-evaluate the first scan's strict geometry conditions.
                    scan = _scan_nominal(
                        env, policy, state,
                        instruction=task["instruction"], target=task["target_object"],
                        support=task["final_support_object"], seed=int(candidate["reset_seed"]),
                        settle_steps=int(task["settle_steps"]), max_steps=int(task["nominal_max_steps"]),
                        release_confirm_steps=int(config["predicates"]["release_confirm_steps"]),
                        stability_dwell_steps=int(config["predicates"]["stability_dwell_steps"]),
                        obstacle=None, geometry=geometry, capture_exact=False,
                    )
                record = _source_scan_record(
                    {**candidate, "candidate_order": order}, state, scan,
                    config["horizons_actions_before_release"],
                )
                if record["mechanically_valid"]:
                    reference = np.asarray(list(_geometry_signature(raw_geometry).values()))
                    current = np.asarray(list(record["geometry_signature"].values()))
                    if not np.allclose(reference, current, rtol=0.0, atol=2e-4):
                        record["mechanically_valid"] = False
                        record["rejection_reason"] = "task_geometry_drift"
                candidate_rows.append(record)
                if record["mechanically_valid"]:
                    passing_payloads.append((record, state, scan))
                _save_gif(
                    staging / "media" / f"candidate_{order:02d}_{'pass' if record['mechanically_valid'] else 'reject'}.gif",
                    scan["frames"], stride=1,
                )
            except Exception as exc:
                candidate_rows.append({
                    **candidate, "candidate_order": order, "mechanically_valid": False,
                    "source_state_sha256": array_sha256(state),
                    "rejection_reason": "candidate_execution_error",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        selected_payloads = passing_payloads[: int(task["select_nominal_successes"])]
        selected_rows = []
        freeze_blocker = None
        if len(selected_payloads) < int(task["select_nominal_successes"]):
            freeze_blocker = (
                f"only {len(selected_payloads)} of the required four predeclared candidates "
                "passed nominal success and mechanical validity"
            )
        if geometry is None or raw_geometry is None:
            freeze_blocker = freeze_blocker or "task geometry could not be resolved"
            # Keep a machine-readable blocked artifact rather than inventing dimensions.
            geometry = {"parameters": [], "blocked": True}

        for selected_slot, (record, state, scan) in enumerate(selected_payloads):
            source_id = str(record["source_id"])
            source_dir = staging / "states" / source_id
            source_dir.mkdir(parents=True)
            state_path = source_dir / "source_state.npy"
            np.save(state_path, state)
            measured = scan["measured_geometry"]
            plate_xy = np.asarray(measured["plate_center_xyz"], dtype=float)[:2]
            try:
                locations = choose_auxiliary_locations(
                    plate_xy=plate_xy,
                    table_bounds_xy=measured["table_bounds_xy"],
                    swept_bounds_xy=scan["swept_bounds_xy"],
                    object_xy=[
                        value[:2] for value in scan["task_object_positions"].values()
                    ],
                    geometry=geometry,
                    config=config,
                )
            except Exception as exc:
                locations = {"mechanically_valid": False, "error": f"{type(exc).__name__}: {exc}"}
                freeze_blocker = freeze_blocker or f"source {source_id} has no valid matched-control/setdown location"
            selected = {
                **record,
                "selected_slot": selected_slot,
                "source_state_path": state_path.relative_to(staging).as_posix(),
                "source_state_file_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                "instruction": task["instruction"],
                "task_suite": task["suite"],
                "task_id": int(task["task_id"]),
                "target_object": task["target_object"],
                "final_support_object": task["final_support_object"],
                "measured_geometry": measured,
                "camera_contract": scan["camera_contract"],
                "task_object_positions_at_source": scan["task_object_positions"],
                "matched_locations": locations,
            }
            selected_rows.append(selected)

        # Selection decisions are now closed.  Nothing below uses intervention outcomes.
        physical_blocks = []
        if not geometry.get("blocked"):
            for source in selected_rows:
                locations = source["matched_locations"]
                if locations.get("mechanically_valid") is False:
                    continue
                plate_xy = np.asarray(
                    source["measured_geometry"]["plate_center_xyz"], dtype=float
                )[:2]
                for parameter in geometry["parameters"]:
                    on_center = plate_xy + np.asarray([
                        float(parameter["patch_center_offset_m"]), 0.0,
                    ])
                    on_patch = build_support_patch(
                        parameter,
                        center_xy=on_center,
                        plate_top_z_m=float(source["measured_geometry"]["plate_top_z_m"]),
                    )
                    offpath_patch = build_support_patch(
                        parameter,
                        center_xy=locations["offpath_patch_center_xy"],
                        plate_top_z_m=float(source["measured_geometry"]["table_top_z_m"]),
                    )
                    stable_xy = np.asarray(locations["stable_offset_target_xy"], dtype=float)
                    stable_support_z = {
                        "on_hazard": max(
                            float(source["measured_geometry"]["plate_top_z_m"]),
                            patch_surface_upper_z(on_patch, stable_xy),
                        ),
                        "offpath": float(source["measured_geometry"]["plate_top_z_m"]),
                        "no_hazard": float(source["measured_geometry"]["plate_top_z_m"]),
                    }
                    half_x, half_y = patch_xy_half_extents(parameter)
                    physical_blocks.append({
                        "source_id": source["source_id"],
                        "source_state_sha256": source["source_state_sha256"],
                        "parameter_id": parameter["parameter_id"],
                        "on_hazard_patch": on_patch,
                        "offpath_patch": offpath_patch,
                        "no_hazard_patch": None,
                        "patch_world_xy_half_extents_m": [half_x, half_y],
                        "plate_support_center_xy": plate_xy.tolist(),
                        "plate_support_radius_m": float(geometry["plate_support_radius_m"]),
                        "strict_success_com_radius_m": float(geometry["strict_success_com_radius_m"]),
                        "catastrophe_escape_radius_m": float(geometry["catastrophe_plate_escape_radius_m"]),
                        "stable_offset_target_xy": stable_xy.tolist(),
                        "stable_offset_support_z_by_condition": stable_support_z,
                        "safe_setdown_target_xyz": [
                            *locations["safe_setdown_target_xy"],
                            float(source["measured_geometry"]["table_top_z_m"])
                            + float(geometry["bowl_half_height_m"]) + 0.002,
                        ],
                        "matched_control_validation": {
                            "offpath_candidate_log": locations["offpath_candidate_log"],
                            "no_hazard_is_absent": True,
                            "identical_patch_except_pose": canonical_sha256({
                                key: value for key, value in on_patch.items()
                                if key not in {"pos"}
                            }) == canonical_sha256({
                                key: value for key, value in offpath_patch.items()
                                if key not in {"pos"}
                            }),
                        },
                        "predicate_tolerances": config["predicates"],
                    })
        frozen_protocol = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "family_id": FAMILY_ID,
            "scientific_question": "Can a mechanically distinct final-support disturbance create same-binary-Base-risk strict option ambiguity?",
            "repository_commit": repo["git_commit"],
            "checkpoint_revision": revision,
            "checkpoint_identity": policy.checkpoint_identity,
            "task": {
                "suite": task["suite"], "task_id": int(task["task_id"]),
                "instruction": task["instruction"], "target_object": task["target_object"],
                "final_support_object": task["final_support_object"],
            },
            "source_selection_rule": task["selection_rule"],
            "complete_candidate_registry_sha256": canonical_sha256(candidate_rows),
            "selected_sources": [{
                "source_id": row["source_id"], "source_state_sha256": row["source_state_sha256"],
                "state_index": row["state_index"], "reset_seed": row["reset_seed"],
                "nominal_release_action_index": row["nominal_release_action_index"],
                "nominal_anchor_indices": row["nominal_anchor_indices"],
                "plate_center_xyz": row["measured_geometry"]["plate_center_xyz"],
                "matched_locations": row["matched_locations"],
            } for row in selected_rows],
            "conditions": list(CONDITIONS),
            "options": list(OPTIONS),
            "horizons_actions_before_release": list(config["horizons_actions_before_release"]),
            "geometry": geometry,
            "physical_blocks": physical_blocks,
            "parameter_contract": config["parameters"],
            "predicate_contract": config["predicates"],
            "controller_contract": config["controller"],
            "restoration_contract": config["restoration"],
            "utility": {"task_success": 1, "safe_noncompletion": 0, "catastrophe": -1},
            "tie_contract": "strict ties remain ties and never count as support",
            "ordinary_option_outcome_row_cap": 324,
            "selection_inputs": "nominal success and mechanical geometry only; no learned scores or router artifacts",
        }
        protocol_sha = canonical_sha256(frozen_protocol)
        final_root = output_parent / f"non_glass_option_ambiguity_screen_{protocol_sha[:12]}_{_path_stamp()}"
        if final_root.exists():
            raise FileExistsError(f"refusing to overwrite existing output root {final_root}")
        staging.rename(final_root)
        staging = final_root

        resolved = copy.deepcopy(config)
        resolved["resolved_geometry"] = geometry
        resolved["resolved_sources"] = selected_rows
        resolved["resolved_physical_blocks"] = physical_blocks
        resolved["checkpoint_revision"] = revision
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "non_glass_unstable_placement_screen_manifest",
            "created_at_utc": _utc_now(),
            "family_id": FAMILY_ID,
            "protocol_sha256": protocol_sha,
            "frozen_protocol": frozen_protocol,
            "repository": repo,
            "runtime_at_freeze": runtime_provenance(),
            "checkpoint_identity": policy.checkpoint_identity,
            "engineering_only": False,
            "freeze_blocker": freeze_blocker,
            "expected_linked_decisions": len(selected_rows) * 27,
            "expected_canonical_decisions": len(selected_rows) * 21,
            "expected_ordinary_option_rows": len(selected_rows) * 81,
            "validated_caps": accounting,
            "phase": "FROZEN",
        }
        _write_json_exclusive(final_root / "screen_manifest.json", manifest)
        _write_json_exclusive(final_root / "resolved_config.json", resolved)
        _write_json_exclusive(final_root / "source_registry.json", {
            "schema_version": SCHEMA_VERSION,
            "selection_closed_before_non_base_outcomes": True,
            "candidate_list_complete": True,
            "candidate_sources": candidate_rows,
            "selected_sources": selected_rows,
            "rejected_sources": [row for row in candidate_rows if not row.get("mechanically_valid", False)],
        })
        _write_json_exclusive(final_root / "future_benchmark_exclusions.json", {
            "schema_version": SCHEMA_VERSION,
            "registry_role": "permanent exclusion of every disposable-screen source and seed from future claim-bearing benchmarks",
            "protocol_sha256": protocol_sha,
            "excluded": [{
                "source_id": row["source_id"],
                "source_state_sha256": row.get("source_state_sha256"),
                "state_index": row["state_index"],
                "reset_seed": row["reset_seed"],
                "task_suite": task["suite"],
                "task_id": int(task["task_id"]),
                "selected_for_option_screen": any(
                    selected["source_id"] == row["source_id"]
                    for selected in selected_rows
                ),
                "reason": "exposed in disposable unstable-final-placement source screen",
            } for row in candidate_rows if row.get("source_state_sha256")],
        })
        attempts = build_attempted_grid(selected_rows, config) if len(selected_rows) == 4 else []
        with (final_root / "attempted_configurations.jsonl").open("x", encoding="utf-8") as handle:
            for row in attempts:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        for name in (
            "mechanical_validity.jsonl", "restoration_audit.jsonl",
            "decision_metadata.jsonl", "option_outcomes.jsonl",
        ):
            _touch_exclusive(final_root / name)
        _write_json_exclusive(final_root / "logs" / "freeze_complete.json", {
            "created_at_utc": _utc_now(), "protocol_sha256": protocol_sha,
            "selected_source_count": len(selected_rows), "freeze_blocker": freeze_blocker,
        })
        print(f"RESULT_ROOT={final_root}", flush=True)
        return final_root
    except Exception:
        if staging.exists():
            (staging / "logs").mkdir(parents=True, exist_ok=True)
            failure = staging / "logs" / "freeze_failure.txt"
            if not failure.exists():
                failure.write_text(traceback.format_exc())
        raise


def _expected_environment_hashes(scan: Mapping[str, Any], index: int) -> dict[str, str]:
    runtime = scan["controller_states"][index]
    return {
        "simulator_state_sha256": array_sha256(scan["states"][index]),
        "controller_state_sha256": _controller_state_sha256(runtime),
        "continuation_state_sha256": _continuation_state_sha256(runtime),
        "observation_sha256": _observation_sha256(scan["observations"][index]),
        "model_xml_sha256": hashlib.sha256(str(scan["model_xml"]).encode("utf-8")).hexdigest(),
    }


def _anchor_invalidity(
    scan: Mapping[str, Any], index: int, *, geometry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Reject a branch state that is unheld, already fallen, tilted, or slipping."""

    xyz = np.asarray(scan["target_xyz_before"][index], dtype=float)
    tilt = float(scan["target_tilt_before"][index])
    relative = np.asarray(scan["target_eef_relative_before"][index], dtype=float)
    prior_index = max(0, int(index) - 2)
    prior_relative = np.asarray(
        scan["target_eef_relative_before"][prior_index], dtype=float
    )
    relative_motion = float(np.linalg.norm(relative - prior_relative))
    relative_limit = (
        float(config["predicates"]["anchor_relative_motion_bowl_radius_fraction"])
        * float(geometry["bowl_footprint_radius_m"])
    )
    reason = None
    if not bool(scan["grasped_before"][index]):
        reason = "branch_anchor_target_not_held"
    elif float(xyz[2]) <= float(geometry["fall_com_height_threshold_m"]):
        reason = "branch_anchor_already_below_fall_threshold"
    elif tilt >= float(config["predicates"]["catastrophe_tilt_deg"]):
        reason = "branch_anchor_already_excessively_tilted"
    elif relative_motion > relative_limit:
        reason = "branch_anchor_target_slipping_relative_to_eef"
    return reason, {
        "target_xyz": xyz.tolist(),
        "target_tilt_deg": tilt,
        "target_eef_relative_xyz": relative.tolist(),
        "relative_motion_over_two_actions_m": relative_motion,
        "relative_motion_limit_m": relative_limit,
        "held": bool(scan["grasped_before"][index]),
    }


def _restore_anchor(
    env: LiberoEnv, policy, *, state: np.ndarray, runtime_state: Mapping[str, np.ndarray],
    rng_state: Mapping[str, np.ndarray], model_xml: str, expected: Mapping[str, str],
    checkpoint_revision: str, instruction: str, label: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    obs = env.reset_to_exact(state, model_xml=model_xml)
    obs = env.restore_controller_state(dict(runtime_state))
    policy.reset()
    _restore_rng_state(rng_state)
    actual_environment = _branch_start_hashes(env, obs)
    _require_branch_start_hashes(actual_environment, expected, label=label)
    rng_sha = _rng_state_sha256(rng_state)
    policy_sha = _policy_continuation_sha256(
        rng_sha256=rng_sha, checkpoint_revision=checkpoint_revision, instruction=instruction,
    )
    actual = {
        **actual_environment,
        "rng_state_sha256": rng_sha,
        "policy_continuation_state_sha256": policy_sha,
    }
    actual["branch_start_sha256"] = _complete_branch_hash(
        actual_environment, rng_sha256=rng_sha, policy_continuation_sha256=policy_sha,
    )
    return obs, actual


def _support_contact(
    *, xyz: np.ndarray, option: str, source: Mapping[str, Any],
    geometry: Mapping[str, Any], patch: Mapping[str, Any] | None,
    tolerance_m: float,
) -> bool:
    bowl_half = float(geometry["bowl_half_height_m"])
    bottom = float(xyz[2]) - bowl_half
    if option == "safe_setdown":
        center = np.asarray(source["matched_locations"]["safe_setdown_target_xy"], dtype=float)
        near = float(np.linalg.norm(xyz[:2] - center)) <= 2.0 * float(geometry["bowl_footprint_radius_m"])
        return bool(
            near and bottom
            <= float(source["measured_geometry"]["table_top_z_m"]) + tolerance_m
        )
    plate = np.asarray(source["measured_geometry"]["plate_center_xyz"], dtype=float)[:2]
    near_plate = float(np.linalg.norm(xyz[:2] - plate)) <= float(geometry["plate_support_radius_m"]) + float(geometry["bowl_footprint_radius_m"])
    surface = float(source["measured_geometry"]["plate_top_z_m"])
    if patch is not None:
        surface = max(surface, patch_surface_upper_z(patch, xyz[:2]))
    return bool(near_plate and bottom <= surface + tolerance_m)


def _make_tracker(
    *, option: str, source: Mapping[str, Any], geometry: Mapping[str, Any],
    config: Mapping[str, Any], initially_held: bool,
) -> StabilityOutcomeTracker:
    predicates = config["predicates"]
    return StabilityOutcomeTracker(
        option=option,
        plate_center_xy=np.asarray(source["measured_geometry"]["plate_center_xyz"], dtype=float)[:2],
        strict_success_radius_m=float(geometry["strict_success_com_radius_m"]),
        catastrophe_escape_radius_m=float(geometry["catastrophe_plate_escape_radius_m"]),
        safe_center_xy=np.asarray(source["matched_locations"]["safe_setdown_target_xy"], dtype=float),
        safe_radius_m=float(predicates["safe_setdown_radius_bowl_radii"]) * float(geometry["bowl_footprint_radius_m"]),
        table_bounds_xy=np.asarray(source["measured_geometry"]["table_bounds_xy"], dtype=float),
        fall_height_threshold_m=float(geometry["fall_com_height_threshold_m"]),
        uncontrolled_displacement_m=float(geometry["uncontrolled_displacement_m"]),
        stable_tilt_deg=float(predicates["stable_tilt_deg"]),
        catastrophe_tilt_deg=float(predicates["catastrophe_tilt_deg"]),
        tilt_sustain_steps=int(predicates["tilt_sustain_steps"]),
        release_confirm_steps=int(predicates["release_confirm_steps"]),
        stability_dwell_steps=int(predicates["stability_dwell_steps"]),
        initially_held=initially_held,
    )


def _controller_for_option(
    *, option: str, obs: Mapping[str, Any], source: Mapping[str, Any],
    geometry: Mapping[str, Any], config: Mapping[str, Any], patch: Mapping[str, Any] | None,
) -> HeldObjectPlacementController:
    target = str(config["task"]["target_object"])
    bowl_half = float(geometry["bowl_half_height_m"])
    if option == "stable_offset_place":
        xy = np.asarray(source["matched_locations"]["stable_offset_target_xy"], dtype=float)
        support_z = float(source["measured_geometry"]["plate_top_z_m"])
        if patch is not None:
            support_z = max(support_z, patch_surface_upper_z(patch, xy))
    elif option == "safe_setdown":
        xy = np.asarray(source["matched_locations"]["safe_setdown_target_xy"], dtype=float)
        support_z = float(source["measured_geometry"]["table_top_z_m"])
    else:
        raise ValueError(option)
    controller = config["controller"]
    bowl_radius = float(geometry["bowl_footprint_radius_m"])
    return HeldObjectPlacementController(
        target_name=target,
        target_xyz=[float(xy[0]), float(xy[1]), support_z + bowl_half + 0.002],
        lift_m=max(float(controller["minimum_lift_m"]), float(controller["lift_bowl_radius_fraction"]) * bowl_radius),
        gain=float(controller["cartesian_gain"]),
        tolerance_m=float(controller["position_tolerance_m"]),
        leg_cap_steps=int(controller["leg_cap_steps"]),
        pre_release_hold_steps=int(controller["pre_release_hold_steps"]),
        release_steps=int(controller["release_steps"]),
        retreat_m=float(controller["retreat_bowl_radius_fraction"]) * bowl_radius,
    )


def _run_branch(
    env: LiberoEnv,
    policy,
    *,
    option: str,
    obs: dict[str, Any],
    branch_hashes: Mapping[str, str],
    instruction: str,
    source: Mapping[str, Any],
    geometry: Mapping[str, Any],
    config: Mapping[str, Any],
    patch: Mapping[str, Any] | None,
    capture_media: bool,
) -> dict[str, Any]:
    target = str(config["task"]["target_object"])
    initially_held = _held(env, obs, target)
    tracker = _make_tracker(
        option=option, source=source, geometry=geometry, config=config,
        initially_held=initially_held,
    )
    if not initially_held:
        return {
            "outcome": None, "utility": None, "invalidity_reason": "branch_anchor_target_not_held",
            "branch_start_sha256": branch_hashes["branch_start_sha256"],
            "branch_start_hashes": dict(branch_hashes), "first_action": None,
            "steps": 0, "trace_sha256": canonical_sha256([]), "trace": [], "frames": [],
        }
    controller = None if option == "base_continue" else _controller_for_option(
        option=option, obs=obs, source=source, geometry=geometry, config=config, patch=patch,
    )
    max_steps = int(config["controller"][{
        "base_continue": "base_max_steps",
        "stable_offset_place": "stable_offset_max_steps",
        "safe_setdown": "safe_setdown_max_steps",
    }[option]])
    frames = [env.render(obs, int(config["execution"]["render_size"]))] if capture_media else []
    first_action = None
    termination = "max_steps"
    for step in range(max_steps):
        if env.episode_terminated():
            termination = "robosuite_episode_horizon"
            break
        if option == "base_continue":
            if tracker.task_success_seen and tracker.release_step is not None:
                action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            else:
                policy_obs = env.policy_observation(obs, policy.resize_size)
                action = np.asarray(policy.act(policy_obs, instruction), dtype=np.float32)
        else:
            assert controller is not None
            action = np.asarray(controller.step(obs), dtype=np.float32)
        if first_action is None:
            first_action = action.copy()
        obs, _, done, _ = env.step(action.tolist())
        xyz = np.asarray(obs[f"{target}_pos"], dtype=float)
        support_contact = _support_contact(
            xyz=xyz, option=option, source=source, geometry=geometry, patch=patch,
            tolerance_m=float(config["predicates"]["support_contact_tolerance_m"]),
        )
        robot_contact = False if patch is None else bool(env.sim_view.max_contact_force(
            list(ROBOT_CONTACT_BODIES), against=[str(patch["name"])],
        ) >= float(config["predicates"]["robot_hazard_contact_force_n"]))
        tracker.update(
            step=step + 1, object_xyz=xyz,
            tilt_deg=float(env.sim_view.object_tilt_deg(target)),
            grasped=_held(env, obs, target), task_success=bool(done),
            support_contact=support_contact, robot_hazard_contact=robot_contact,
        )
        if capture_media and (
            step % int(config["execution"]["media_stride"]) == 0 or tracker.ready_to_stop
        ):
            frames.append(env.render(obs, int(config["execution"]["render_size"])))
        if tracker.ready_to_stop:
            termination = "frozen_terminal_contract"
            break
        if env.episode_terminated():
            termination = "robosuite_episode_horizon"
            break
    result = tracker.finalize()
    result.update({
        "branch_start_sha256": branch_hashes["branch_start_sha256"],
        "branch_start_hashes": dict(branch_hashes),
        "first_action": None if first_action is None else first_action.tolist(),
        "first_action_sha256": None if first_action is None else array_sha256(first_action),
        "steps": len(tracker.trace),
        "termination": termination,
        "controller_final_stage": None if controller is None else controller.i,
        "frames": frames,
    })
    return result


def _write_trace(root: Path, decision_id: str, label: str, result: Mapping[str, Any]) -> str:
    path = root / "logs" / "traces" / decision_id / f"{label}.json"
    _write_json_exclusive(path, {
        "outcome": result.get("outcome"), "invalidity_reason": result.get("invalidity_reason"),
        "catastrophe_reason": result.get("catastrophe_reason"),
        "trace_sha256": result.get("trace_sha256"), "trace": result.get("trace", []),
    })
    return path.relative_to(root).as_posix()


def _attempt_links(attempts: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    links: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        links.setdefault(str(row["canonical_decision_id"]), []).append(dict(row))
    return links


def _write_option_links(
    path: Path, links: Sequence[Mapping[str, Any]], option: str,
    result: Mapping[str, Any], *, trace_path: str | None,
) -> None:
    for link in links:
        row = {
            "schema_version": SCHEMA_VERSION,
            "family_id": FAMILY_ID,
            **dict(link),
            "option": option,
            "outcome": result.get("outcome"),
            "utility": result.get("utility"),
            "invalidity_reason": result.get("invalidity_reason"),
            "catastrophe_reason": result.get("catastrophe_reason"),
            "branch_start_sha256": result.get("branch_start_sha256"),
            "branch_start_hashes": result.get("branch_start_hashes"),
            "first_action": result.get("first_action"),
            "first_action_sha256": result.get("first_action_sha256"),
            "steps": int(result.get("steps", 0)),
            "termination": result.get("termination"),
            "object_release_observed": result.get("object_release_observed"),
            "stability_dwell_observed_steps": result.get("stability_dwell_observed_steps"),
            "trace_sha256": result.get("trace_sha256"),
            "trace_path": trace_path,
        }
        _append_jsonl(path, row)


def _null_result(reason: str, branch_start: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "outcome": None, "utility": None, "invalidity_reason": reason,
        "catastrophe_reason": None,
        "branch_start_sha256": None if branch_start is None else branch_start.get("branch_start_sha256"),
        "branch_start_hashes": None if branch_start is None else dict(branch_start),
        "first_action": None, "first_action_sha256": None, "steps": 0,
        "termination": "not_executed_fail_closed", "object_release_observed": None,
        "stability_dwell_observed_steps": 0, "trace_sha256": canonical_sha256([]),
    }


def _load_attempts(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def execute_screen(result_root: str | Path, *, checkpoint_revision: str) -> dict[str, Any]:
    root = Path(result_root).resolve()
    manifest_path = root / "screen_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{root} is not a frozen screen root")
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((root / "resolved_config.json").read_text())
    source_registry = json.loads((root / "source_registry.json").read_text())
    revision = require_checkpoint_revision(checkpoint_revision)
    if revision != manifest["frozen_protocol"]["checkpoint_revision"]:
        raise ValueError("execute checkpoint revision differs from the frozen protocol")
    if manifest.get("freeze_blocker"):
        _write_json_exclusive(root / "execution_complete.json", {
            "created_at_utc": _utc_now(), "status": "blocked_at_freeze",
            "reason": manifest["freeze_blocker"],
        })
        return analyze_screen(root)
    if manifest.get("engineering_only"):
        raise ValueError("engineering mock roots cannot enter real execution")
    for name in ("mechanical_validity.jsonl", "restoration_audit.jsonl", "decision_metadata.jsonl", "option_outcomes.jsonl"):
        if (root / name).stat().st_size:
            raise FileExistsError(f"refusing to overwrite or resume non-empty {root / name}")
    _write_json_exclusive(root / "execution_started.json", {
        "created_at_utc": _utc_now(), "protocol_sha256": manifest["protocol_sha256"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })
    repo = repository_provenance(ROOT, require_clean=True)
    if repo["git_commit"] != manifest["repository"]["git_commit"]:
        raise RuntimeError("execution commit differs from the frozen protocol commit")

    from crashbench.policies import OpenVLAPolicy

    policy = OpenVLAPolicy(
        pretrained_checkpoint=config["execution"]["checkpoint"],
        checkpoint_revision=revision,
        unnorm_key=config["execution"]["unnorm_key"],
        center_crop=True,
        capture_hidden=False,
    )
    env = LiberoEnv(
        config["task"]["suite"], int(config["task"]["task_id"]),
        seed=int(config["execution"]["rollout_seed"]),
    )
    geometry = config["resolved_geometry"]
    attempts = _load_attempts(root / "attempted_configurations.jsonl")
    links_by_id = _attempt_links(attempts)
    selected_sources = source_registry["selected_sources"]
    physical_by_key = {
        (str(row["source_state_sha256"]), str(row["parameter_id"])): row
        for row in config["resolved_physical_blocks"]
    }
    media_index = []
    completed_canonical = 0

    for source_index, source in enumerate(selected_sources):
        source_state_path = root / source["source_state_path"]
        source_state = np.load(source_state_path, allow_pickle=False)
        if array_sha256(source_state) != source["source_state_sha256"]:
            raise RuntimeError(f"source hash mismatch for {source['source_id']}")
        for parameter_id in SEVERITY_IDS:
            physical = physical_by_key[(source["source_state_sha256"], parameter_id)]
            patches = {
                "on_hazard": physical["on_hazard_patch"],
                "offpath": physical["offpath_patch"],
                "no_hazard": None,
            }
            for condition in CONDITIONS:
                if condition == "no_hazard" and parameter_id != SEVERITY_IDS[0]:
                    continue
                patch = patches[condition]
                scan_error = None
                scan = None
                try:
                    scan = _scan_nominal(
                        env, policy, source_state,
                        instruction=config["task"]["instruction"],
                        target=config["task"]["target_object"],
                        support=config["task"]["final_support_object"],
                        seed=int(source["reset_seed"]),
                        settle_steps=int(config["task"]["settle_steps"]),
                        max_steps=int(config["task"]["nominal_max_steps"]),
                        release_confirm_steps=int(config["predicates"]["release_confirm_steps"]),
                        stability_dwell_steps=0,
                        obstacle=patch, geometry=geometry, capture_exact=True,
                    )
                    if scan["camera_contract"]["sha256"] != source["camera_contract"]["sha256"]:
                        scan_error = "camera_or_preprocessing_contract_changed"
                    elif scan["initial_robot_hazard_force_n"] >= float(config["predicates"]["initial_contact_force_n"]):
                        scan_error = "hazard_initial_robot_contact_or_penetration"
                    elif scan["initial_target_hazard_contact"]:
                        scan_error = "hazard_initial_target_contact_or_penetration"
                    elif canonical_sha256(scan["task_object_positions"]) != canonical_sha256(
                        source["task_object_positions_at_source"]
                    ):
                        scan_error = "non_hazard_scene_object_layout_changed"
                    elif scan["release_action_index"] is None:
                        scan_error = "condition_has_no_nominal_release_event"
                except Exception as exc:
                    scan_error = f"condition_scan_error:{type(exc).__name__}:{exc}"

                scan_dir = root / "states" / source["source_id"] / "condition_scans" / f"{parameter_id}__{condition}"
                scan_dir.mkdir(parents=True, exist_ok=True)
                if scan is not None:
                    (scan_dir / "model.xml").write_text(scan["model_xml"])
                    _write_json_exclusive(scan_dir / "scan_summary.json", {
                        "release_action_index": scan["release_action_index"],
                        "camera_contract": scan["camera_contract"],
                        "model_xml_sha256": scan["model_xml_sha256"],
                        "initial_robot_hazard_force_n": scan["initial_robot_hazard_force_n"],
                        "initial_target_hazard_contact": scan["initial_target_hazard_contact"],
                        "error": scan_error,
                    })
                    if source_index == 0:
                        media_name = _save_gif(
                            root / "media" / f"scan_{parameter_id}_{condition}.gif",
                            scan["frames"], stride=1,
                        )
                        media_index.append({"kind": "condition_scan", "parameter_id": parameter_id, "condition": condition, "file": media_name})

                for horizon in config["horizons_actions_before_release"]:
                    canonical_parameter = "shared" if condition == "no_hazard" else parameter_id
                    canonical_id = canonical_sha256({
                        "source_state_sha256": source["source_state_sha256"],
                        "parameter_id": canonical_parameter,
                        "horizon_actions": int(horizon),
                        "condition": condition,
                    })
                    links = links_by_id[canonical_id]
                    if scan_error is not None or scan is None:
                        reason = scan_error or "condition_scan_unavailable"
                        _append_jsonl(root / "mechanical_validity.jsonl", {
                            "schema_version": SCHEMA_VERSION,
                            "canonical_decision_id": canonical_id,
                            "source_state_sha256": source["source_state_sha256"],
                            "parameter_id": canonical_parameter,
                            "horizon_actions": int(horizon), "condition": condition,
                            "pre_intervention_eligible": False, "eligible": False,
                            "reason": reason,
                        })
                        for option in OPTIONS:
                            _write_option_links(root / "option_outcomes.jsonl", links, option, _null_result(reason), trace_path=None)
                        continue
                    anchor_index = int(scan["release_action_index"]) - int(horizon) + 1
                    if anchor_index < 0 or anchor_index >= len(scan["states"]):
                        reason = "release_before_predeclared_horizon"
                        _append_jsonl(root / "mechanical_validity.jsonl", {
                            "schema_version": SCHEMA_VERSION, "canonical_decision_id": canonical_id,
                            "source_state_sha256": source["source_state_sha256"],
                            "parameter_id": canonical_parameter, "horizon_actions": int(horizon),
                            "condition": condition, "pre_intervention_eligible": False,
                            "eligible": False, "reason": reason,
                        })
                        for option in OPTIONS:
                            _write_option_links(root / "option_outcomes.jsonl", links, option, _null_result(reason), trace_path=None)
                        continue
                    expected = _expected_environment_hashes(scan, anchor_index)
                    rng_state = scan["rng_states"][anchor_index]
                    rng_sha = _rng_state_sha256(rng_state)
                    policy_sha = _policy_continuation_sha256(
                        rng_sha256=rng_sha, checkpoint_revision=revision,
                        instruction=config["task"]["instruction"],
                    )
                    expected_complete = _complete_branch_hash(
                        expected, rng_sha256=rng_sha, policy_continuation_sha256=policy_sha,
                    )
                    anchor_reason, anchor_diagnostics = _anchor_invalidity(
                        scan, anchor_index, geometry=geometry, config=config,
                    )
                    anchor_dir = scan_dir / f"H{int(horizon)}"
                    anchor_dir.mkdir()
                    np.save(anchor_dir / "simulator_state.npy", scan["states"][anchor_index])
                    np.savez_compressed(anchor_dir / "continuation_state.npz", **scan["controller_states"][anchor_index])
                    np.savez_compressed(anchor_dir / "rng_state.npz", **rng_state)
                    metadata = {
                        "schema_version": SCHEMA_VERSION,
                        "family_id": FAMILY_ID,
                        "canonical_decision_id": canonical_id,
                        "canonical_evidence": True,
                        "source_id": source["source_id"],
                        "source_state_sha256": source["source_state_sha256"],
                        "parameter_id": canonical_parameter,
                        "linked_parameter_ids": [row["parameter_id"] for row in links],
                        "condition": condition,
                        "horizon_actions": int(horizon),
                        "nominal_release_action_index": int(scan["release_action_index"]),
                        "anchor_index": anchor_index,
                        "branch_state_path": (anchor_dir / "simulator_state.npy").relative_to(root).as_posix(),
                        "continuation_state_path": (anchor_dir / "continuation_state.npz").relative_to(root).as_posix(),
                        "rng_state_path": (anchor_dir / "rng_state.npz").relative_to(root).as_posix(),
                        "model_xml_path": (scan_dir / "model.xml").relative_to(root).as_posix(),
                        "expected_branch_start_hashes": {
                            **expected, "rng_state_sha256": rng_sha,
                            "policy_continuation_state_sha256": policy_sha,
                            "branch_start_sha256": expected_complete,
                        },
                        "first_nominal_action": scan["nominal_actions"][anchor_index].tolist(),
                        "first_nominal_action_sha256": array_sha256(scan["nominal_actions"][anchor_index]),
                        "task_identity_unchanged": True,
                        "instruction_unchanged": env.task_description == config["task"]["instruction"],
                        "camera_preprocessing_unchanged": scan["camera_contract"]["sha256"] == source["camera_contract"]["sha256"],
                        "matched_control_links": [row["parameter_id"] for row in links],
                        "anchor_mechanical_diagnostics": anchor_diagnostics,
                        "anchor_mechanical_invalidity": anchor_reason,
                    }
                    _append_jsonl(root / "decision_metadata.jsonl", metadata)

                    frozen_branch_start = {
                        **expected,
                        "rng_state_sha256": rng_sha,
                        "policy_continuation_state_sha256": policy_sha,
                        "branch_start_sha256": expected_complete,
                    }
                    if anchor_reason is not None:
                        _append_jsonl(root / "mechanical_validity.jsonl", {
                            "schema_version": SCHEMA_VERSION,
                            "canonical_decision_id": canonical_id,
                            "source_state_sha256": source["source_state_sha256"],
                            "parameter_id": canonical_parameter,
                            "horizon_actions": int(horizon),
                            "condition": condition,
                            "anchor_mechanical_diagnostics": anchor_diagnostics,
                            "pre_intervention_eligible": False,
                            "eligible": False,
                            "reason": anchor_reason,
                        })
                        for option in OPTIONS:
                            _write_option_links(
                                root / "option_outcomes.jsonl", links, option,
                                _null_result(anchor_reason, branch_start=frozen_branch_start),
                                trace_path=None,
                            )
                        continue

                    def restored(label: str) -> tuple[dict[str, Any], dict[str, str]]:
                        return _restore_anchor(
                            env, policy,
                            state=scan["states"][anchor_index],
                            runtime_state=scan["controller_states"][anchor_index],
                            rng_state=rng_state, model_xml=scan["model_xml"], expected=expected,
                            checkpoint_revision=revision, instruction=config["task"]["instruction"],
                            label=f"{canonical_id}:{label}",
                        )

                    audit_error = None
                    base_results = []
                    try:
                        for repeat in range(2):
                            obs, hashes = restored(f"base_repeat_{repeat + 1}")
                            result = _run_branch(
                                env, policy, option="base_continue", obs=obs,
                                branch_hashes=hashes, instruction=config["task"]["instruction"],
                                source=source, geometry=geometry, config=config, patch=patch,
                                capture_media=False,
                            )
                            result["trace_path"] = _write_trace(root, canonical_id, f"base_repeat_{repeat + 1}", result)
                            base_results.append(result)
                    except Exception as exc:
                        audit_error = f"base_repeat_execution_error:{type(exc).__name__}:{exc}"
                    audit_passed = False
                    max_trace_difference = None
                    first_action_matches_scan = False
                    if audit_error is None and len(base_results) == 2:
                        expected_action = np.asarray(scan["nominal_actions"][anchor_index], dtype=np.float32)
                        audit_result = evaluate_base_repeat_audit(
                            expected_first_action=expected_action,
                            repeat_results=base_results,
                            expected_branch_start_sha256=expected_complete,
                            first_action_atol=float(config["restoration"]["first_action_atol"]),
                            trace_atol=float(config["restoration"]["trace_atol"]),
                        )
                        first_action_matches_scan = bool(audit_result["first_actions_match"])
                        max_trace_difference = float(audit_result["max_trace_abs_difference"])
                        audit_passed = bool(audit_result["passed"])
                        if not audit_passed:
                            audit_error = f"base_repeat_restoration_mismatch:{audit_result['reason']}"
                    _append_jsonl(root / "restoration_audit.jsonl", {
                        "schema_version": SCHEMA_VERSION,
                        "canonical_decision_id": canonical_id,
                        "source_state_sha256": source["source_state_sha256"],
                        "parameter_id": canonical_parameter,
                        "condition": condition, "horizon_actions": int(horizon),
                        "attempted": True, "passed": audit_passed,
                        "expected_branch_start_sha256": expected_complete,
                        "base_repeat_branch_start_sha256": [row["branch_start_sha256"] for row in base_results],
                        "first_action_matches_frozen_nominal": first_action_matches_scan,
                        "terminal_outcomes": [row.get("outcome") for row in base_results],
                        "trace_sha256": [row.get("trace_sha256") for row in base_results],
                        "max_trace_abs_difference": max_trace_difference,
                        "trace_atol": float(config["restoration"]["trace_atol"]),
                        "error": audit_error,
                    })
                    pre_eligible = bool(
                        audit_passed and scan["grasped_before"][anchor_index]
                        and base_results[0].get("invalidity_reason") is None
                    )
                    if condition == "on_hazard" and base_results:
                        base_catastrophe = base_results[0].get("outcome") == "catastrophe"
                        support_instability = base_results[0].get("catastrophe_reason") in {
                            None,
                            "object_fell_below_geometry_threshold",
                            "sustained_excessive_tilt",
                            "uncontrolled_post_release_displacement",
                            "com_left_valid_plate_support",
                            "object_left_table_bounds",
                        }
                        pre_eligible = pre_eligible and (not base_catastrophe or support_instability)

                    option_results: dict[str, dict[str, Any]] = {}
                    if base_results:
                        option_results["base_continue"] = base_results[0]
                    else:
                        option_results["base_continue"] = _null_result(audit_error or "base_repeat_unavailable")
                    post_invalidity = None
                    if pre_eligible:
                        for option in ("stable_offset_place", "safe_setdown"):
                            try:
                                obs, hashes = restored(option)
                                result = _run_branch(
                                    env, policy, option=option, obs=obs,
                                    branch_hashes=hashes, instruction=config["task"]["instruction"],
                                    source=source, geometry=geometry, config=config, patch=patch,
                                    capture_media=(source_index == 0 and condition == "on_hazard" and int(horizon) == 10),
                                )
                                result["trace_path"] = _write_trace(root, canonical_id, option, result)
                                if result.get("invalidity_reason"):
                                    post_invalidity = str(result["invalidity_reason"])
                                if result.get("frames"):
                                    name = f"branch_{source['source_id']}_{parameter_id}_{condition}_H{horizon}_{option}.gif"
                                    media = _save_gif(root / "media" / name, result["frames"])
                                    media_index.append({
                                        "kind": "branch", "canonical_decision_id": canonical_id,
                                        "option": option, "file": media,
                                    })
                                option_results[option] = result
                            except Exception as exc:
                                post_invalidity = f"option_execution_error:{option}:{type(exc).__name__}:{exc}"
                                option_results[option] = _null_result(post_invalidity, branch_start={
                                    **expected, "rng_state_sha256": rng_sha,
                                    "policy_continuation_state_sha256": policy_sha,
                                    "branch_start_sha256": expected_complete,
                                })
                    else:
                        reason = audit_error or "pre_intervention_mechanical_ineligibility"
                        for option in ("stable_offset_place", "safe_setdown"):
                            option_results[option] = _null_result(reason, branch_start={
                                **expected, "rng_state_sha256": rng_sha,
                                "policy_continuation_state_sha256": policy_sha,
                                "branch_start_sha256": expected_complete,
                            })
                    final_eligible = bool(pre_eligible and post_invalidity is None)
                    validity_reason = None if final_eligible else (
                        post_invalidity or audit_error or "pre_intervention_mechanical_ineligibility"
                    )
                    offpath_geometry_valid = condition != "offpath" or any(
                        bool(row.get("selected"))
                        and bool(row.get("outside_nominal_swept_volume"))
                        and bool(row.get("outside_final_support"))
                        for row in source["matched_locations"]["offpath_candidate_log"]
                    )
                    _append_jsonl(root / "mechanical_validity.jsonl", {
                        "schema_version": SCHEMA_VERSION,
                        "canonical_decision_id": canonical_id,
                        "source_state_sha256": source["source_state_sha256"],
                        "parameter_id": canonical_parameter,
                        "condition": condition, "horizon_actions": int(horizon),
                        "task_identity_unchanged": True,
                        "source_and_instruction_unchanged": True,
                        "camera_preprocessing_unchanged": True,
                        "no_initial_robot_hazard_contact": scan["initial_robot_hazard_force_n"] < float(config["predicates"]["initial_contact_force_n"]),
                        "no_initial_target_hazard_contact": not scan["initial_target_hazard_contact"],
                        "anchor_target_held": bool(scan["grasped_before"][anchor_index]),
                        "on_hazard_failure_is_support_instability_not_path_collision": condition != "on_hazard" or option_results["base_continue"].get("invalidity_reason") != "pre_support_robot_hazard_contact",
                        "offpath_outside_support_and_swept_volume": offpath_geometry_valid,
                        "matched_no_hazard_linked": True,
                        "pre_intervention_eligible": pre_eligible,
                        "eligible": final_eligible,
                        "reason": validity_reason,
                    })
                    for option in OPTIONS:
                        result = option_results[option]
                        trace_path = result.get("trace_path")
                        _write_option_links(
                            root / "option_outcomes.jsonl", links, option, result,
                            trace_path=trace_path,
                        )
                    completed_canonical += 1
                    _append_jsonl(root / "logs" / "execution_progress.jsonl", {
                        "created_at_utc": _utc_now(), "event": "canonical_decision_complete",
                        "canonical_decision_id": canonical_id, "completed_canonical": completed_canonical,
                    })

    _write_json_exclusive(root / "media" / "media_index.json", {
        "schema_version": SCHEMA_VERSION, "items": media_index,
    })
    _write_json_exclusive(root / "slurm_provenance.json", {
        "created_at_utc": _utc_now(), "git_commit": repo["git_commit"],
        "protocol_sha256": manifest["protocol_sha256"],
        "checkpoint_revision": revision,
        "rollout_seed": config["execution"]["rollout_seed"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT", "p33100"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "gengpu"),
        "hostname": os.uname().nodename,
    })
    _write_json_exclusive(root / "execution_complete.json", {
        "created_at_utc": _utc_now(), "status": "complete",
        "canonical_decisions_completed": completed_canonical,
        "ordinary_option_outcome_rows": sum(1 for line in (root / "option_outcomes.jsonl").read_text().splitlines() if line.strip()),
    })
    return analyze_screen(root)


def _write_mock_run(*, output_parent: str | Path, config_path: str | Path) -> Path:
    config = json.loads(Path(config_path).read_text())
    accounting = validate_screen_config(config)
    sources = [{
        "source_id": f"mock_source_{index}",
        "source_state_sha256": hashlib.sha256(f"mock_source_{index}".encode()).hexdigest(),
        "state_index": index, "reset_seed": 9000 + index,
    } for index in range(4)]
    protocol = {
        "schema_version": SCHEMA_VERSION, "family_id": FAMILY_ID,
        "engineering_only": True, "sources": sources,
        "conditions": list(CONDITIONS), "options": list(OPTIONS),
        "horizons": config["horizons_actions_before_release"],
        "parameters": config["parameters"], "utility": {"task_success": 1, "safe_noncompletion": 0, "catastrophe": -1},
    }
    protocol_sha = canonical_sha256(protocol)
    root = create_immutable_result_root(output_parent, protocol_sha, _path_stamp())
    (root / "logs").mkdir()
    (root / "media").mkdir()
    (root / "states").mkdir()
    _write_json_exclusive(root / "screen_manifest.json", {
        "schema_version": SCHEMA_VERSION, "kind": "non_glass_unstable_placement_screen_manifest",
        "created_at_utc": _utc_now(), "family_id": FAMILY_ID,
        "protocol_sha256": protocol_sha, "frozen_protocol": protocol,
        "engineering_only": True, "freeze_blocker": None,
        "expected_linked_decisions": 108, "expected_canonical_decisions": 84,
        "expected_ordinary_option_rows": 324, "validated_caps": accounting,
        "repository": {"git_commit": "engineering-mock"}, "phase": "ENGINEERING_COMPLETE",
    })
    _write_json_exclusive(root / "resolved_config.json", config)
    _write_json_exclusive(root / "source_registry.json", {
        "candidate_list_complete": True, "candidate_sources": sources,
        "selected_sources": sources, "rejected_sources": [],
    })
    _write_json_exclusive(root / "future_benchmark_exclusions.json", {
        "registry_role": "engineering mock only; hashes are synthetic and cannot enter science",
        "excluded": sources,
    })
    attempts = build_attempted_grid(sources, config)
    with (root / "attempted_configurations.jsonl").open("x") as handle:
        for row in attempts:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    for name in ("mechanical_validity.jsonl", "restoration_audit.jsonl", "decision_metadata.jsonl", "option_outcomes.jsonl"):
        _touch_exclusive(root / name)
    links = _attempt_links(attempts)
    for decision_id, group in sorted(links.items()):
        canonical = next(row for row in group if row["canonical_evidence"])
        parameter = canonical["parameter_id"]
        condition = canonical["condition"]
        branch_hash = hashlib.sha256(f"branch:{decision_id}".encode()).hexdigest()
        _append_jsonl(root / "mechanical_validity.jsonl", {
            "canonical_decision_id": decision_id, "eligible": True, "reason": None,
            "source_state_sha256": canonical["source_state_sha256"],
        })
        _append_jsonl(root / "restoration_audit.jsonl", {
            "canonical_decision_id": decision_id, "passed": True,
            "source_state_sha256": canonical["source_state_sha256"],
        })
        _append_jsonl(root / "decision_metadata.jsonl", {
            **canonical, "anchor_index": 42, "canonical_evidence": True,
        })
        if condition == "on_hazard" and parameter == "moderate":
            outcomes = {"base_continue": "catastrophe", "stable_offset_place": "task_success", "safe_setdown": "safe_noncompletion"}
        elif condition == "on_hazard" and parameter == "severe":
            outcomes = {"base_continue": "catastrophe", "stable_offset_place": "catastrophe", "safe_setdown": "safe_noncompletion"}
        else:
            outcomes = {"base_continue": "task_success", "stable_offset_place": "safe_noncompletion", "safe_setdown": "safe_noncompletion"}
        for option in OPTIONS:
            result = {
                "outcome": outcomes[option], "utility": {"task_success": 1, "safe_noncompletion": 0, "catastrophe": -1}[outcomes[option]],
                "invalidity_reason": None, "catastrophe_reason": None,
                "branch_start_sha256": branch_hash,
                "branch_start_hashes": {"branch_start_sha256": branch_hash},
                "first_action": [0.0] * 7, "first_action_sha256": hashlib.sha256(b"mock").hexdigest(),
                "steps": 24, "termination": "engineering_mock",
                "object_release_observed": True, "stability_dwell_observed_steps": 24,
                "trace_sha256": hashlib.sha256(f"trace:{decision_id}:{option}".encode()).hexdigest(),
            }
            _write_option_links(root / "option_outcomes.jsonl", group, option, result, trace_path=None)
    _write_json_exclusive(root / "execution_complete.json", {
        "created_at_utc": _utc_now(), "status": "engineering_mock_complete",
    })
    analyze_screen(root)
    print(f"RESULT_ROOT={root}", flush=True)
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", required=True, help="result parent for freeze; exact frozen root for execute-only")
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--screen-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mock", action="store_true", help="engineering-only synthetic plumbing check; never scientific evidence")
    args = parser.parse_args()
    if not args.freeze_only and not args.execute:
        raise SystemExit("select --freeze-only, --execute, or both")
    if args.mock and args.execute and not args.freeze_only:
        raise SystemExit("--mock execute requires --freeze-only in the same invocation")
    result_root = None
    if args.freeze_only:
        result_root = freeze_protocol(
            output_parent=args.output, config_path=args.screen_config,
            checkpoint_revision=args.checkpoint_revision, mock=args.mock,
        )
    if args.execute and not args.mock:
        result_root = Path(args.output).resolve() if result_root is None else result_root
        decision = execute_screen(result_root, checkpoint_revision=args.checkpoint_revision)
        print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
        print(f"RESULT_ROOT={result_root}", flush=True)


if __name__ == "__main__":
    main()
