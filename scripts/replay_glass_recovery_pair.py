#!/usr/bin/env python3
"""Replay one collected glass branch without loading OpenVLA.

This verifies that saved executed actions, exact start state, scene definition,
and declared outcome still agree.  It is intended both for dataset audit and for
quick simulator-only debugging on Quest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import (
    PairedTrajectoryRecord,
    array_sha256,
    read_placement_manifest,
    read_trajectory_manifest,
    validate_episode_arrays,
)
from crashbench.predicates import build_any, build_predicate
from crashbench.scenario import PredicateSpec
from scripts.collect_glass_recovery_pairs import (
    _controller_state_sha256,
    _glass_predicate_specs,
    _partition_scene,
)


def accepted_pair_dir_from_manifest(manifest: str | Path) -> Path:
    """Resolve the first accepted pair directory recorded in a split manifest."""

    manifest = Path(manifest)
    records = read_trajectory_manifest(manifest)
    pair_id = records[0].pair_id
    pair_records = [record for record in records if record.pair_id == pair_id]
    pair_dirs = {
        (manifest.parent / record.arrays_path).parent.resolve()
        for record in pair_records
    }
    if len(pair_dirs) != 1:
        raise ValueError(f"{pair_id} arrays span multiple directories: {sorted(pair_dirs)}")
    pair_dir = pair_dirs.pop()
    if not (pair_dir / "pair.json").is_file():
        raise FileNotFoundError(f"accepted pair metadata missing: {pair_dir / 'pair.json'}")
    return pair_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", required=True)
    pair_source = parser.add_mutually_exclusive_group(required=True)
    pair_source.add_argument("--pair-dir")
    pair_source.add_argument(
        "--manifest",
        help="select the first accepted pair recorded in this split manifest",
    )
    parser.add_argument("--branch", choices=(
        "nominal_catastrophe", "oracle_recovery", "off_path_control", "blocked_safe_abort",
    ), required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pair_dir = (
        accepted_pair_dir_from_manifest(args.manifest)
        if args.manifest else Path(args.pair_dir)
    )
    pair_payload = json.loads((pair_dir / "pair.json").read_text())
    records = [PairedTrajectoryRecord.from_dict(row) for row in pair_payload["records"]]
    record = next(row for row in records if row.trajectory_kind == args.branch)
    placements, _ = read_placement_manifest(args.placements)
    placement = next(row for row in placements if row.placement_id == record.placement_id)

    if args.branch in {"nominal_catastrophe", "oracle_recovery"}:
        glasses = [placement.on_path_glass]
        state_path = pair_dir / "precrash_onpath_state.npy"
    elif args.branch == "off_path_control":
        glasses = [placement.off_path_glass]
        state_path = pair_dir / "offpath_start_state.npy"
    else:
        glasses = placement.blocked_glasses
        state_path = pair_dir / "blocked_start_state.npy"
    state = np.load(state_path)
    if array_sha256(state) != record.branch_start_state_sha256:
        raise SystemExit(f"start-state hash mismatch for {state_path}")
    controller_path = pair_dir / "controller_state.npz"
    controller_state = dict(np.load(controller_path))
    expected_controller_hash = record.metadata.get("controller_state_sha256")
    if _controller_state_sha256(controller_state) != expected_controller_hash:
        raise SystemExit(f"controller-state hash mismatch for {controller_path}")

    arrays_path = pair_dir.parents[1] / record.arrays_path
    arrays = dict(np.load(arrays_path))
    validate_episode_arrays(arrays, record.n_steps)
    env = LiberoEnv(placement.task_suite, placement.task_id)
    obstacles, movables = _partition_scene(glasses)
    env.reset_to_exact(state, obstacles=obstacles, movable_objects=movables)
    env.restore_controller_state(controller_state)
    crash = build_any(_glass_predicate_specs(glasses))
    success = build_predicate(PredicateSpec("libero_task_success", {}))
    crashed = succeeded = False
    event_step = None
    for index, action in enumerate(arrays["executed_action"]):
        env.step(np.asarray(action, dtype=float).tolist())
        if crash(env.sim_view):
            crashed, event_step = True, index
            break
        if success(env.sim_view):
            succeeded, event_step = True, index
            break
    safe_abort = not crashed and not succeeded and args.branch == "blocked_safe_abort"
    observed = {
        "crashed": crashed,
        "succeeded": succeeded,
        "safe_abort": safe_abort,
        "event_step": event_step,
        "peak_robot_contact_force_n": float(env.sim_view.peak_force),
    }
    expected = {
        "crashed": record.crashed,
        "succeeded": record.succeeded,
        "safe_abort": record.safe_abort,
    }
    matches = all(observed[key] == value for key, value in expected.items())
    payload = {
        "schema_version": 1,
        "pair_id": record.pair_id,
        "branch": args.branch,
        "expected": expected,
        "observed": observed,
        "matches": matches,
    }
    print(json.dumps(payload, indent=2), flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    if not matches:
        raise SystemExit("replay outcome does not match manifest")


if __name__ == "__main__":
    main()
