#!/usr/bin/env python3
"""Run nominal-only D2/D3 source preflight for one frozen mechanism/task cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.source_registry import ExposureRegistry, SourceIdentity
from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import array_sha256
from crashbench.policies import build_policy
from scripts.expansion.hash_tree_manifest import resolve_git_head


PI0_CHECKPOINT = "gs://openpi-assets/checkpoints/pi0_libero"
ALLOWED_MECHANISMS = (
    "fragile_path_collision_v2",
    "observation_staleness_v1",
    "action_drift_v1",
    "narrow_clearance_v1",
)
CELL_ORDER = tuple((mechanism, task_id) for mechanism in ALLOWED_MECHANISMS for task_id in (0, 2))


def cell_for_array_index(index: int) -> tuple[str, int]:
    try:
        return CELL_ORDER[int(index)]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"array index must be in 0..{len(CELL_ORDER) - 1}") from exc


def select_attempts(plan: dict[str, Any], *, mechanism_id: str, task_id: int) -> list[dict[str, Any]]:
    if mechanism_id not in ALLOWED_MECHANISMS:
        raise ValueError(
            f"mechanism {mechanism_id!r} is not authorized for nominal preflight; "
            "unstable_final_placement_v2 requires separate user authorization"
        )
    rows = [
        row
        for row in plan.get("attempts", [])
        if row.get("mechanism_id") == mechanism_id and int(row.get("task_id", -1)) == task_id
    ]
    rows.sort(key=lambda row: int(row["candidate_index"]))
    if len(rows) != 8 or [int(row["candidate_index"]) for row in rows] != list(range(8)):
        raise ValueError(f"frozen cell {mechanism_id}/task{task_id} must contain indices 0..7")
    return rows


def _action_hash(action: np.ndarray) -> str:
    return array_sha256(np.asarray(action))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--exposure-registry", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--task-id", type=int, choices=(0, 2), required=True)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-policy-steps", type=int, default=220)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite nominal preflight: {args.output}")
    plan = json.loads(args.source_plan.read_text())
    attempts = select_attempts(plan, mechanism_id=args.mechanism_id, task_id=args.task_id)
    registry = ExposureRegistry.load(args.exposure_registry)
    for row in attempts:
        reasons = registry.exposure_reasons(SourceIdentity(reset_seed=row["reset_seed"]))
        if not reasons:
            raise RuntimeError(f"attempt is not exposed before policy load: {row['attempt_id']}")

    env = LiberoEnv("libero_spatial", args.task_id, model_family="pi0", seed=0)
    init_states = env.default_init_states()
    policy = build_policy(
        "pi0",
        pretrained_checkpoint=PI0_CHECKPOINT,
        config_name="pi0_libero",
        num_open_loop_steps=5,
    )
    rows = []
    for attempt in attempts:
        started = time.monotonic()
        candidate_index = int(attempt["candidate_index"])
        reset_seed = int(attempt["reset_seed"])
        init_state = np.asarray(init_states[candidate_index])
        source_hash = array_sha256(init_state)
        record: dict[str, Any] = {
            "attempt_id": attempt["attempt_id"],
            "mechanism_id": args.mechanism_id,
            "task_id": args.task_id,
            "candidate_index": candidate_index,
            "reset_seed": reset_seed,
            "source_state_sha256": source_hash,
            "source_role": "EXPOSED_ENGINEERING_SCREEN",
            "option_outcomes_opened": 0,
        }
        try:
            env.seed(reset_seed)
            policy.reset()
            obs = env.reset_to(init_state)
            for _ in range(args.settle_steps):
                obs, _, _, _ = env.step(env.dummy_action())
            eef_path = []
            action_hashes = []
            success = False
            termination = "max_policy_steps"
            for step in range(args.max_policy_steps):
                eef_path.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist())
                policy_obs = env.policy_observation(obs, policy.resize_size)
                action = np.asarray(policy.act(policy_obs, env.task_description))
                action_hashes.append(_action_hash(action))
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    success = True
                    termination = "task_success"
                    break
                if env.episode_terminated():
                    termination = "environment_horizon"
                    break
            record.update(
                {
                    "status": "NOMINAL_COMPLETE",
                    "task_success": success,
                    "termination": termination,
                    "policy_steps": len(action_hashes),
                    "eef_path_xyz": eef_path,
                    "action_sha256": action_hashes,
                    "eef_path_length_m": float(
                        np.sum(np.linalg.norm(np.diff(np.asarray(eef_path), axis=0), axis=1))
                    ) if len(eef_path) >= 2 else 0.0,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "NOMINAL_OPERATIONAL_FAILURE",
                    "failure": f"{type(exc).__name__}:{exc}",
                    "task_success": False,
                    "termination": "exception",
                    "policy_steps": 0,
                    "eef_path_xyz": [],
                    "action_sha256": [],
                    "eef_path_length_m": 0.0,
                }
            )
        record["elapsed_seconds"] = time.monotonic() - started
        rows.append(record)

    complete = sum(row["status"] == "NOMINAL_COMPLETE" for row in rows)
    successes = sum(bool(row["task_success"]) for row in rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_nominal_preflight_cell",
        "mechanism_id": args.mechanism_id,
        "task_id": args.task_id,
        "source_plan": str(args.source_plan),
        "source_plan_sha256": hashlib.sha256(args.source_plan.read_bytes()).hexdigest(),
        "execution": {
            "git_commit": resolve_git_head(Path.cwd()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "policy": "pi0",
            "checkpoint": PI0_CHECKPOINT,
        },
        "attempted_sources": len(rows),
        "complete_sources": complete,
        "nominal_successes": successes,
        "nominal_success_rate": successes / len(rows),
        "all_attempts_accounted": len(rows) == 8,
        "option_outcomes_opened": 0,
        "rows": rows,
    }
    payload["cell_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "complete": complete, "successes": successes}, sort_keys=True))
    if complete != 8:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
