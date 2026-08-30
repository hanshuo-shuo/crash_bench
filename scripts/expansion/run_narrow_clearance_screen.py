#!/usr/bin/env python3
"""Run one frozen-source narrow-clearance D2 option screen shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.branching.engine import BranchEngine
from crashbench.branching.state import capture_exact_state
from crashbench.envs import LiberoEnv
from crashbench.mechanisms.narrow_clearance import CorridorGeometry
from crashbench.options.common import BacktrackRequeryOption, SafeStopOption
from crashbench.policies import build_policy
from scripts.expansion.hash_tree_manifest import resolve_git_head
from scripts.expansion.run_fragile_screen import (
    anchor_steps_from_nominal_path,
    classify_terminal,
    screen_utility,
)


PI0_CHECKPOINT = "gs://openpi-assets/checkpoints/pi0_libero"
OPTIONS = ("base_continue", "backtrack_requery", "safe_stop")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanical-cell", type=Path, required=True)
    parser.add_argument("--nominal-cell", type=Path, required=True)
    parser.add_argument("--source-index", type=int, choices=range(8), required=True)
    parser.add_argument("--max-branch-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite narrow screen shard: {args.output}")
    mechanical = json.loads(args.mechanical_cell.read_text())
    nominal = json.loads(args.nominal_cell.read_text())
    if mechanical.get("mechanism_id") != "narrow_clearance_v1":
        raise ValueError("narrow screen requires narrow-clearance mechanical cell")
    source = mechanical["rows"][args.source_index]
    nominal_source = nominal["rows"][args.source_index]
    if source["attempt_id"] != nominal_source["attempt_id"]:
        raise ValueError("mechanical and nominal source attempt mismatch")
    if source["status"] != "MECHANICAL_VALID":
        raise ValueError("narrow source is not mechanically valid")
    task_id = int(source["task_id"])
    reset_seed = int(source["reset_seed"])
    init_index = int(source["candidate_index"])
    env = LiberoEnv("libero_spatial", task_id, model_family="pi0", seed=reset_seed)
    init_state = np.asarray(env.default_init_states()[init_index])
    policy = build_policy(
        "pi0", pretrained_checkpoint=PI0_CHECKPOINT,
        config_name="pi0_libero", num_open_loop_steps=5
    )
    engine = BranchEngine(env, policy)
    blocks = []
    for severity_id, severity in sorted(source["severities"].items()):
        narrow = severity["conditions"]["narrow"]
        anchor_steps = anchor_steps_from_nominal_path(
            nominal_source["eef_path_xyz"], narrow["center_xy"]
        )
        for condition, geometry in sorted(severity["conditions"].items()):
            block_id = f"{source['attempt_id']}:{severity_id}:{condition}"
            corridor = None if geometry is None else CorridorGeometry(**geometry)
            obstacles = [] if corridor is None else corridor.obstacles()
            wall_names = [row["name"] for row in obstacles]
            env.seed(reset_seed)
            policy.reset()
            obs = env.reset_to(init_state, obstacles=obstacles)
            for _ in range(10):
                obs, _, _, _ = env.step(env.dummy_action())
            eef_history = []
            last_gripper = -1.0
            preanchor_invalid = None
            for _ in range(anchor_steps):
                eef_history.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy())
                policy_obs = env.policy_observation(obs, policy.resize_size)
                action = np.asarray(policy.act(policy_obs, env.task_description))
                last_gripper = float(action[6])
                obs, _, done, _ = env.step(action.tolist())
                force = float(
                    env.sim_view.max_contact_force(env.sim_view._robot_bodies(), against=wall_names)
                ) if wall_names else 0.0
                if force >= 75.0:
                    preanchor_invalid = "wall_catastrophe_before_anchor"
                    break
                if done or env.episode_terminated():
                    preanchor_invalid = "episode_terminated_before_anchor"
                    break
            if preanchor_invalid or len(eef_history) < 4:
                blocks.append(
                    {
                        "block_id": block_id,
                        "severity_id": severity_id,
                        "condition": condition,
                        "status": "PRE_ANCHOR_MECHANICAL_INVALID",
                        "reason": preanchor_invalid or "insufficient_eef_history",
                        "option_outcomes": [],
                    }
                )
                continue
            bundle = capture_exact_state(
                env,
                policy,
                identity={
                    "source_id": source["source_state_sha256"],
                    "policy_id": "pi0",
                    "mechanism_id": "narrow_clearance_v1",
                    "task_id": f"libero_spatial:{task_id}",
                    "trajectory_id": source["attempt_id"],
                    "anchor_id": f"{severity_id}:{condition}:step{anchor_steps}",
                },
                provenance={
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "source_role": "EXPOSED_ENGINEERING_SCREEN",
                    "wall_names": wall_names,
                },
                declared_branch_seed=0,
            )

            def rollout(option_id: str):
                def run(current_env, current_policy):
                    current_obs = dict(current_env.sim_view._obs)
                    path = [np.asarray(current_obs["robot0_eef_pos"], dtype=np.float64)]
                    catastrophe = False
                    success = False
                    steps = 0
                    if option_id == "backtrack_requery":
                        option = BacktrackRequeryOption(history_offset=3)
                        start_view = option.spec.view(
                            {
                                "eef_history": np.asarray(eef_history),
                                "current_eef": np.asarray(current_obs["robot0_eef_pos"]),
                                "force_history": [0.0],
                                "gripper_command": last_gripper,
                            }
                        )
                        option.start(start_view)
                        handoff = False
                    elif option_id == "safe_stop":
                        option = SafeStopOption(stable_force_threshold=75.0, stable_steps=3)
                        force_history = [0.0]
                        option.start(option.spec.view({"force_history": force_history, "gripper_command": last_gripper}))
                    else:
                        option = None
                    for step in range(args.max_branch_steps):
                        if option_id == "safe_stop":
                            force_history.append(0.0)
                            view = option.spec.view(
                                {"force_history": force_history, "gripper_command": last_gripper}
                            )
                            action = option.step(view)
                        elif option_id == "backtrack_requery" and not handoff:
                            view = option.spec.view(
                                {
                                    "eef_history": np.asarray(eef_history),
                                    "current_eef": np.asarray(current_obs["robot0_eef_pos"]),
                                    "force_history": [0.0],
                                    "gripper_command": last_gripper,
                                }
                            )
                            if option.termination_reason(view).value in {"hazard_clear", "max_duration"}:
                                current_policy.reset()
                                handoff = True
                                policy_obs = current_env.policy_observation(current_obs, current_policy.resize_size)
                                action = np.asarray(current_policy.act(policy_obs, current_env.task_description))
                            else:
                                action = option.step(view)
                        else:
                            policy_obs = current_env.policy_observation(current_obs, current_policy.resize_size)
                            action = np.asarray(current_policy.act(policy_obs, current_env.task_description))
                        current_obs, _, done, _ = current_env.step(np.asarray(action).tolist())
                        steps = step + 1
                        path.append(np.asarray(current_obs["robot0_eef_pos"], dtype=np.float64))
                        force = float(
                            current_env.sim_view.max_contact_force(
                                current_env.sim_view._robot_bodies(), against=wall_names
                            )
                        ) if wall_names else 0.0
                        catastrophe = force >= 75.0
                        if catastrophe or done or current_env.episode_terminated():
                            success = bool(done)
                            break
                        if option_id == "safe_stop" and option.termination_reason(view).value != "running":
                            break
                    path_length = float(
                        np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1))
                    ) if len(path) >= 2 else 0.0
                    return classify_terminal(
                        task_success=success,
                        catastrophe=catastrophe,
                        intervention=option_id != "base_continue",
                        steps=steps,
                        path_length=path_length,
                    )
                return run

            try:
                branch_rows = engine.execute_complete_lattice(
                    bundle,
                    {option_id: rollout(option_id) for option_id in OPTIONS},
                    branch_seeds=[0],
                )
                outcomes = [
                    {
                        "option_id": row.option_id,
                        "branch_start_bundle_id": row.branch_start_bundle_id,
                        "branch_start_component_hashes": dict(row.branch_start_component_hashes),
                        "terminal_signature": row.terminal_signature,
                        "outcome": dict(row.outcome),
                        "utility": screen_utility(row.outcome),
                    }
                    for row in branch_rows
                ]
                blocks.append(
                    {
                        "block_id": block_id,
                        "severity_id": severity_id,
                        "condition": condition,
                        "status": "COMPLETE_REALIZED_OPTIONS",
                        "bundle_id": bundle.bundle_id,
                        "option_outcomes": outcomes,
                    }
                )
            except Exception as exc:
                blocks.append(
                    {
                        "block_id": block_id,
                        "severity_id": severity_id,
                        "condition": condition,
                        "status": "OPERATIONAL_FAILURE",
                        "reason": f"{type(exc).__name__}:{exc}",
                        "option_outcomes": [],
                    }
                )
    complete = sum(block["status"] == "COMPLETE_REALIZED_OPTIONS" for block in blocks)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_narrow_screen_source_shard",
        "mechanism_id": "narrow_clearance_v1",
        "task_id": task_id,
        "source_index": args.source_index,
        "attempt_id": source["attempt_id"],
        "source_state_sha256": source["source_state_sha256"],
        "execution": {
            "git_commit": resolve_git_head(Path.cwd()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "planned_blocks": 9,
        "complete_blocks": complete,
        "all_blocks_accounted": len(blocks) == 9,
        "option_outcomes_opened": sum(len(block["option_outcomes"]) for block in blocks),
        "blocks": blocks,
    }
    payload["shard_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "complete_blocks": complete}, sort_keys=True))


if __name__ == "__main__":
    main()
