#!/usr/bin/env python3
"""Collect one train/calibration/development staleness source shard for D5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.branching.artifacts import ContentAddressedStore, pack_numeric_mapping
from crashbench.branching.engine import BranchEngine
from crashbench.branching.state import capture_exact_state
from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import array_sha256
from crashbench.mechanisms.observation_staleness import ObservationDelayQueue
from crashbench.options.common import ObservationRefreshOption, SafeStopOption
from crashbench.policies import build_policy
from scripts.expansion.hash_tree_manifest import resolve_git_head
from scripts.expansion.run_fragile_screen import classify_terminal, screen_utility
from scripts.expansion.run_staleness_screen import mechanism_observation, queue_state_sha256


PI0_CHECKPOINT = "gs://openpi-assets/checkpoints/pi0_libero"
ALLOWED_ROLES = ("train", "calibration", "development")
ROLE_ORDER = {role: index for index, role in enumerate(ALLOWED_ROLES)}
ANCHOR_STEPS = (5, 10, 15)
SEVERITIES = {"delay_1": 1, "delay_3": 3, "delay_5": 5}
CONDITIONS = ("stale", "fresh_control", "matched_buffer_control")
OPTIONS = ("base_continue", "observation_refresh", "safe_stop")


def active_assignments(split_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in split_manifest.get("assignments", []) if row.get("role") in ALLOWED_ROLES]
    rows.sort(
        key=lambda row: (
            str(row["task_id"]),
            ROLE_ORDER[str(row["role"])],
            str(row["physical_source_id"]),
        )
    )
    if len(rows) != 48:
        raise ValueError(f"D5 active split must contain 48 train/cal/dev sources, got {len(rows)}")
    if any(row["role"] not in ALLOWED_ROLES for row in rows):
        raise RuntimeError("test role leaked into D5 active assignments")
    return rows


def source_by_physical_id(formal_analysis: dict[str, Any], physical_source_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in formal_analysis.get("selected_sources", [])
        if row.get("physical_source_id") == physical_source_id
    ]
    if len(matches) != 1:
        raise ValueError(f"formal source lookup expected one row, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-analysis", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--assignment-index", type=int, choices=range(48), required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--max-branch-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D5 source shard: {args.output}")
    formal = json.loads(args.formal_analysis.read_text())
    split = json.loads(args.split_manifest.read_text())
    assignment = active_assignments(split)[args.assignment_index]
    if assignment["role"] not in ALLOWED_ROLES:
        raise RuntimeError(f"D5 refuses non-active role: {assignment['role']}")
    source = source_by_physical_id(formal, assignment["physical_source_id"])
    if source["attempt_id"] != assignment["formal_attempt_id"]:
        raise ValueError("formal attempt identity differs between source and split")
    state = np.asarray(source["source_state"], dtype=np.dtype(source["source_state_dtype"]))
    if array_sha256(state) != assignment["source_state_sha256"]:
        raise ValueError("formal source state bytes do not match split identity")
    task_id = int(source["task_id"])
    reset_seed = int(source["reset_seed"])
    env = LiberoEnv("libero_spatial", task_id, model_family="pi0", seed=reset_seed)
    policy = build_policy(
        "pi0", pretrained_checkpoint=PI0_CHECKPOINT,
        config_name="pi0_libero", num_open_loop_steps=5
    )
    engine = BranchEngine(env, policy)
    store = ContentAddressedStore(args.artifact_store)
    blocks = []
    for anchor_steps in ANCHOR_STEPS:
        for severity_id, delay_steps in SEVERITIES.items():
            for condition in CONDITIONS:
                block_id = (
                    f"{assignment['policy_source_id']}:{anchor_steps}:"
                    f"{severity_id}:{condition}"
                )
                queue = ObservationDelayQueue(0 if condition == "fresh_control" else delay_steps)
                env.seed(reset_seed)
                policy.reset()
                obs = env.reset_to_exact(state)
                for _ in range(10):
                    obs, _, _, _ = env.step(env.dummy_action())
                preanchor_invalid = None
                for _ in range(anchor_steps):
                    fresh = env.policy_observation(obs, policy.resize_size)
                    policy_obs = mechanism_observation(queue, fresh, condition=condition)
                    action = np.asarray(policy.act(policy_obs, env.task_description))
                    obs, _, done, _ = env.step(action.tolist())
                    if done or env.episode_terminated():
                        preanchor_invalid = "episode_terminated_before_anchor"
                        break
                if preanchor_invalid:
                    blocks.append(
                        {
                            "block_id": block_id,
                            "anchor_steps": anchor_steps,
                            "severity_id": severity_id,
                            "condition": condition,
                            "status": "PRE_ANCHOR_MECHANICAL_INVALID",
                            "reason": preanchor_invalid,
                            "option_outcomes": [],
                        }
                    )
                    continue
                queue_snapshot = queue.snapshot_state()
                queue_hash = queue_state_sha256(queue_snapshot)
                feature_queue = ObservationDelayQueue(queue.delay_steps)
                feature_queue.restore_state(queue_snapshot)
                fresh_anchor = env.policy_observation(obs, policy.resize_size)
                delivered_anchor = mechanism_observation(
                    feature_queue, fresh_anchor, condition=condition
                )
                feature_ref = store.put_bytes(pack_numeric_mapping(delivered_anchor))
                bundle = capture_exact_state(
                    env,
                    policy,
                    identity={
                        "source_id": assignment["physical_source_id"],
                        "policy_id": "pi0",
                        "mechanism_id": "observation_staleness_v1",
                        "task_id": assignment["task_id"],
                        "trajectory_id": assignment["policy_source_id"],
                        "anchor_id": f"step{anchor_steps}:{severity_id}:{condition}",
                    },
                    provenance={
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "split_role": assignment["role"],
                        "mechanism_queue_sha256": queue_hash,
                    },
                    declared_branch_seed=0,
                )

                def rollout(option_id: str):
                    def run(current_env, current_policy):
                        queue.restore_state(queue_snapshot)
                        current_obs = dict(current_env.sim_view._obs)
                        path = [np.asarray(current_obs["robot0_eef_pos"], dtype=np.float64)]
                        catastrophe = False
                        success = False
                        steps = 0
                        refreshed = False
                        if option_id == "safe_stop":
                            option = SafeStopOption(stable_force_threshold=75.0, stable_steps=3)
                            force_history = [0.0]
                            option.start(option.spec.view({"force_history": force_history, "gripper_command": -1.0}))
                        elif option_id == "observation_refresh":
                            option = ObservationRefreshOption()
                        else:
                            option = None
                        for step in range(args.max_branch_steps):
                            fresh = current_env.policy_observation(current_obs, current_policy.resize_size)
                            if option_id == "safe_stop":
                                force_history.append(
                                    float(current_env.sim_view.max_contact_force(current_env.sim_view._robot_bodies()))
                                )
                                view = option.spec.view(
                                    {"force_history": force_history, "gripper_command": -1.0}
                                )
                                action = option.step(view)
                            else:
                                if option_id == "observation_refresh" and not refreshed:
                                    view = option.spec.view(
                                        {"policy_observation": fresh, "observation_queue": queue}
                                    )
                                    policy_obs = option.refresh(view)
                                    current_policy.reset()
                                    refreshed = True
                                else:
                                    policy_obs = mechanism_observation(queue, fresh, condition=condition)
                                action = np.asarray(current_policy.act(policy_obs, current_env.task_description))
                            current_obs, _, done, _ = current_env.step(np.asarray(action).tolist())
                            steps = step + 1
                            path.append(np.asarray(current_obs["robot0_eef_pos"], dtype=np.float64))
                            force = float(
                                current_env.sim_view.max_contact_force(current_env.sim_view._robot_bodies())
                            )
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
                            "mechanism_queue_sha256": queue_hash,
                            "terminal_signature": row.terminal_signature,
                            "outcome": dict(row.outcome),
                            "utility": screen_utility(row.outcome),
                        }
                        for row in branch_rows
                    ]
                    blocks.append(
                        {
                            "block_id": block_id,
                            "anchor_steps": anchor_steps,
                            "severity_id": severity_id,
                            "condition": condition,
                            "status": "COMPLETE_REALIZED_OPTIONS",
                            "bundle_id": bundle.bundle_id,
                            "mechanism_queue_sha256": queue_hash,
                            "anchor_feature_blob": asdict(feature_ref),
                            "option_outcomes": outcomes,
                        }
                    )
                except Exception as exc:
                    blocks.append(
                        {
                            "block_id": block_id,
                            "anchor_steps": anchor_steps,
                            "severity_id": severity_id,
                            "condition": condition,
                            "status": "OPERATIONAL_FAILURE",
                            "reason": f"{type(exc).__name__}:{exc}",
                            "anchor_feature_blob": asdict(feature_ref),
                            "option_outcomes": [],
                        }
                    )
    complete = sum(block["status"] == "COMPLETE_REALIZED_OPTIONS" for block in blocks)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d5_staleness_source_shard",
        "protocol_sha256": split["protocol_sha256"],
        "physical_source_id": assignment["physical_source_id"],
        "mechanism_source_id": assignment["mechanism_source_id"],
        "policy_source_id": assignment["policy_source_id"],
        "task_id": assignment["task_id"],
        "split_role": assignment["role"],
        "assignment_index": args.assignment_index,
        "execution": {
            "git_commit": resolve_git_head(Path.cwd()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "planned_blocks": 27,
        "complete_blocks": complete,
        "all_blocks_accounted": len(blocks) == 27,
        "test_rows_read": 0,
        "blocks": blocks,
    }
    payload["shard_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "role": assignment["role"], "complete": complete}, sort_keys=True))


if __name__ == "__main__":
    main()
