#!/usr/bin/env python3
"""Capture successful no-glass source traces for E15 candidate generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES
from crashbench.glass_recovery_data import array_sha256
from crashbench.provenance import repository_provenance, require_checkpoint_revision
from scripts.prepare_glass_recovery_placements import _file_sha256


def _robot_body_xyz(env: LiberoEnv) -> np.ndarray:
    _model, data = env.sim_view._live_mj()
    rows = [
        np.asarray(data.xpos[env.sim_view._body_id(_model, name)], dtype=np.float64).copy()
        for name in ROBOT_CONTACT_BODIES
    ]
    return np.stack(rows)


def capture(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_revision = require_checkpoint_revision(args.checkpoint_revision)
    repo = repository_provenance(Path(__file__).resolve().parents[1], require_clean=True)
    declared = os.environ.get("CB_CODE_COMMIT")
    if declared is not None and declared != repo["git_commit"]:
        raise SystemExit("CB_CODE_COMMIT does not match checked-out source")

    from crashbench.policies import OpenVLAPolicy

    env = LiberoEnv(args.suite, args.task_id, seed=args.rollout_seed)
    policy = OpenVLAPolicy(
        pretrained_checkpoint=args.checkpoint,
        checkpoint_revision=checkpoint_revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        capture_hidden=False,
    )
    if args.random_initializations is None:
        states = np.asarray(env.default_init_states())
        indices = list(range(len(states))) if args.state_indices is None else args.state_indices
        invalid = sorted(set(indices) - set(range(len(states))))
        if invalid:
            raise SystemExit(f"state indices outside LIBERO state set: {invalid}")
        source_states = [(index, np.asarray(states[index]), None) for index in indices]
    else:
        source_states = []
        for source_index in range(args.random_initializations):
            reset_seed = args.random_initialization_seed + source_index
            env.seed(reset_seed)
            env.env.reset()
            source_states.append((source_index, env.flat_state(), reset_seed))
    traces = []
    for source_index, source_state, reset_seed in source_states:
        state = np.asarray(source_state, dtype=np.float64)
        env.seed(args.rollout_seed)
        policy.reset()
        obs = env.reset_to(state)
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(env.dummy_action())
        eef_xyz = []
        robot_body_xyz = []
        actions = []
        succeeded = False
        for _step in range(args.max_steps):
            eef_xyz.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy())
            robot_body_xyz.append(_robot_body_xyz(env))
            policy_obs = env.policy_observation(obs, policy.resize_size)
            action = np.asarray(policy.act(policy_obs, env.task_description), dtype=np.float32)
            actions.append(action)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                succeeded = True
                break
        trace_root = output / "traces" / f"state_{source_index:03d}"
        trace_root.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "eef_xyz": np.asarray(eef_xyz, dtype=np.float64),
            "robot_body_xyz": np.asarray(robot_body_xyz, dtype=np.float64),
            "actions": np.asarray(actions, dtype=np.float32),
        }
        row = {
            "task_suite": args.suite,
            "task_id": args.task_id,
            "source_state_index": int(source_index),
            "source_state_sha256": array_sha256(state),
            "task_succeeded": succeeded,
            "steps": len(actions),
            "rollout_seed": args.rollout_seed,
            "source_initialization": (
                "libero_default_init_state" if reset_seed is None else "random_reset"
            ),
            "source_reset_seed": reset_seed,
        }
        state_path = trace_root / "source_state.npy"
        np.save(state_path, state)
        row["source_state_path"] = state_path.relative_to(output).as_posix()
        row["source_state_file_sha256"] = _file_sha256(state_path)
        for name, value in artifacts.items():
            path = trace_root / f"{name}.npy"
            np.save(path, value)
            row[f"{name}_path"] = path.relative_to(output).as_posix()
            row[f"{name}_sha256"] = _file_sha256(path)
        traces.append(row)
        print(
            f"state={source_index} success={succeeded} steps={len(actions)}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "kind": "glass_recovery_nominal_source_traces",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": repo["git_commit"],
        "task_suite": args.suite,
        "task_id": args.task_id,
        "checkpoint_identity": policy.checkpoint_identity,
        "checkpoint_revision": checkpoint_revision,
        "unnorm_key": args.unnorm_key,
        "rollout_seed": args.rollout_seed,
        "source_initialization": (
            "libero_default_init_states"
            if args.random_initializations is None else "random_resets"
        ),
        "random_initialization_seed": (
            None if args.random_initializations is None
            else args.random_initialization_seed
        ),
        "settle_steps": args.settle_steps,
        "max_steps": args.max_steps,
        "robot_body_order": list(ROBOT_CONTACT_BODIES),
        "successful_source_states": sum(row["task_succeeded"] for row in traces),
        "traces": traces,
    }
    manifest = output / "source_traces.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--state-indices", type=int, nargs="+")
    parser.add_argument("--random-initializations", type=int)
    parser.add_argument("--random-initialization-seed", type=int, default=20260817)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.random_initializations is not None and args.state_indices is not None:
        raise SystemExit("random initializations and state indices are mutually exclusive")
    if args.random_initializations is not None and args.random_initializations < 1:
        raise SystemExit("random initializations must be positive")
    if args.max_steps < 1 or args.settle_steps < 0:
        raise SystemExit("max steps must be positive and settle steps nonnegative")
    capture(args)


if __name__ == "__main__":
    main()
