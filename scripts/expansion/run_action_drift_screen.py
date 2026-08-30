#!/usr/bin/env python3
"""Run one frozen-source action-drift D2 option screen shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.branching.engine import BranchEngine
from crashbench.branching.state import capture_exact_state
from crashbench.envs import LiberoEnv
from crashbench.mechanisms.action_drift import ActionDriftInjector
from crashbench.options.common import CorrectiveRequeryOption, SafeStopOption
from crashbench.policies import build_policy
from scripts.expansion.hash_tree_manifest import resolve_git_head
from scripts.expansion.run_fragile_screen import classify_terminal, screen_utility


PI0_CHECKPOINT = "gs://openpi-assets/checkpoints/pi0_libero"
OPTIONS = ("base_continue", "corrective_requery", "safe_stop")


def injector_state_sha256(injector: ActionDriftInjector) -> str:
    return hashlib.sha256(
        json.dumps(asdict(injector.snapshot_state()), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanical-cell", type=Path, required=True)
    parser.add_argument("--nominal-cell", type=Path, required=True)
    parser.add_argument("--source-index", type=int, choices=range(8), required=True)
    parser.add_argument("--anchor-steps", type=int, default=10)
    parser.add_argument("--max-branch-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite action-drift screen shard: {args.output}")
    mechanical = json.loads(args.mechanical_cell.read_text())
    nominal = json.loads(args.nominal_cell.read_text())
    if mechanical.get("mechanism_id") != "action_drift_v1":
        raise ValueError("action-drift screen requires action-drift mechanical cell")
    source = mechanical["rows"][args.source_index]
    nominal_source = nominal["rows"][args.source_index]
    if source["attempt_id"] != nominal_source["attempt_id"]:
        raise ValueError("mechanical and nominal source attempt mismatch")
    if source["status"] != "MECHANICAL_VALID":
        raise ValueError("action-drift source is not mechanically valid")
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
        for condition, parameters in sorted(severity["conditions"].items()):
            block_id = f"{source['attempt_id']}:{severity_id}:{condition}"
            injector = ActionDriftInjector(
                np.asarray(parameters["bias"], dtype=np.float64),
                action_low=np.full(7, -1.0),
                action_high=np.full(7, 1.0),
            )
            env.seed(reset_seed)
            policy.reset()
            obs = env.reset_to(init_state)
            for _ in range(10):
                obs, _, _, _ = env.step(env.dummy_action())
            commanded_history = []
            executed_history = []
            preanchor_invalid = None
            for _ in range(args.anchor_steps):
                policy_obs = env.policy_observation(obs, policy.resize_size)
                commanded = np.asarray(policy.act(policy_obs, env.task_description), dtype=np.float64)
                executed = injector.apply(commanded)
                commanded_history.append(commanded.copy())
                executed_history.append(executed.copy())
                obs, _, done, _ = env.step(executed.tolist())
                force = float(env.sim_view.max_contact_force(env.sim_view._robot_bodies()))
                if force >= 75.0:
                    preanchor_invalid = "catastrophe_before_anchor"
                    break
                if done or env.episode_terminated():
                    preanchor_invalid = "episode_terminated_before_anchor"
                    break
            if preanchor_invalid or len(commanded_history) < 2:
                blocks.append(
                    {
                        "block_id": block_id,
                        "severity_id": severity_id,
                        "condition": condition,
                        "status": "PRE_ANCHOR_MECHANICAL_INVALID",
                        "reason": preanchor_invalid or "insufficient_command_history",
                        "option_outcomes": [],
                    }
                )
                continue
            injector_snapshot = injector.snapshot_state()
            injector_hash = injector_state_sha256(injector)
            bundle = capture_exact_state(
                env,
                policy,
                identity={
                    "source_id": source["source_state_sha256"],
                    "policy_id": "pi0",
                    "mechanism_id": "action_drift_v1",
                    "task_id": f"libero_spatial:{task_id}",
                    "trajectory_id": source["attempt_id"],
                    "anchor_id": f"{severity_id}:{condition}:step{args.anchor_steps}",
                },
                provenance={
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "source_role": "EXPOSED_ENGINEERING_SCREEN",
                    "injector_state_sha256": injector_hash,
                },
                declared_branch_seed=0,
            )

            def rollout(option_id: str):
                def run(current_env, current_policy):
                    injector.restore_state(injector_snapshot)
                    current_obs = dict(current_env.sim_view._obs)
                    path = [np.asarray(current_obs["robot0_eef_pos"], dtype=np.float64)]
                    catastrophe = False
                    success = False
                    steps = 0
                    if option_id == "corrective_requery":
                        option = CorrectiveRequeryOption(max_correction=0.15)
                        start_view = option.spec.view(
                            {
                                "commanded_action_history": np.asarray(commanded_history),
                                "executed_action_history": np.asarray(executed_history),
                                "base_proposed_action": np.asarray(commanded_history[-1]),
                            }
                        )
                        option.start(start_view)
                    elif option_id == "safe_stop":
                        option = SafeStopOption(stable_force_threshold=75.0, stable_steps=3)
                        force_history = [0.0]
                        option.start(option.spec.view({"force_history": force_history, "gripper_command": -1.0}))
                    else:
                        option = None
                    for step in range(args.max_branch_steps):
                        if option_id == "safe_stop":
                            force_history.append(
                                float(current_env.sim_view.max_contact_force(current_env.sim_view._robot_bodies()))
                            )
                            view = option.spec.view(
                                {"force_history": force_history, "gripper_command": -1.0}
                            )
                            commanded = option.step(view)
                        else:
                            policy_obs = current_env.policy_observation(current_obs, current_policy.resize_size)
                            base_action = np.asarray(current_policy.act(policy_obs, current_env.task_description))
                            if option_id == "corrective_requery":
                                view = option.spec.view(
                                    {
                                        "commanded_action_history": np.asarray(commanded_history),
                                        "executed_action_history": np.asarray(executed_history),
                                        "base_proposed_action": base_action,
                                    }
                                )
                                commanded = option.step(view)
                            else:
                                commanded = base_action
                        executed = injector.apply(np.asarray(commanded))
                        current_obs, _, done, _ = current_env.step(executed.tolist())
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
                        "injector_state_sha256": injector_hash,
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
                        "injector_state_sha256": injector_hash,
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
        "kind": "crashbench_expansion_action_drift_screen_source_shard",
        "mechanism_id": "action_drift_v1",
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
