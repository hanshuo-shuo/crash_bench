#!/usr/bin/env python3
"""Collect one shard of fresh, formal nominal source candidates."""

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

from crashbench.data.source_registry import ExposureRegistry, SourceIdentity
from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import array_sha256
from crashbench.policies import build_policy
from scripts.expansion.hash_tree_manifest import resolve_git_head


PI0_CHECKPOINT = "gs://openpi-assets/checkpoints/pi0_libero"
SHARDS = tuple(
    (task_id, start, min(start + 21, 63))
    for task_id in (0, 2)
    for start in (0, 21, 42)
)


def shard_spec(index: int) -> tuple[int, int, int]:
    try:
        return SHARDS[int(index)]
    except (IndexError, ValueError) as exc:
        raise ValueError("formal nominal shard index must be 0..5") from exc


def scene_fingerprint(observation: dict[str, Any]) -> str:
    rows = {}
    for key, value in observation.items():
        if not key.endswith("_pos") or key.startswith("robot0_") or "eef" in key:
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (3,):
            rows[key] = np.round(array, 3).tolist()
    if not rows:
        raise ValueError("fresh reset observation contains no object position fields")
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--exposure-registry", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, choices=range(6), required=True)
    parser.add_argument("--max-policy-steps", type=int, default=220)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite formal nominal shard: {args.output}")
    task_id, start, end = shard_spec(args.shard_index)
    plan = json.loads(args.source_plan.read_text())
    attempts = sorted(
        [row for row in plan["attempts"] if int(row["task_id"]) == task_id],
        key=lambda row: int(row["candidate_index"]),
    )[start:end]
    if len(attempts) != 21 or [int(row["candidate_index"]) for row in attempts] != list(range(start, end)):
        raise ValueError("formal nominal shard does not match frozen 21-source range")
    registry = ExposureRegistry.load(args.exposure_registry)
    for row in attempts:
        if not registry.exposure_reasons(SourceIdentity(reset_seed=row["reset_seed"])):
            raise RuntimeError(f"formal attempt is not pre-exposed: {row['attempt_id']}")
    env = LiberoEnv("libero_spatial", task_id, model_family="pi0", seed=0)
    policy = build_policy(
        "pi0", pretrained_checkpoint=PI0_CHECKPOINT,
        config_name="pi0_libero", num_open_loop_steps=5
    )
    rows = []
    for attempt in attempts:
        reset_seed = int(attempt["reset_seed"])
        record: dict[str, Any] = {
            "attempt_id": attempt["attempt_id"],
            "task_id": task_id,
            "candidate_index": int(attempt["candidate_index"]),
            "reset_seed": reset_seed,
            "source_role": "EXPOSED_FORMAL_NOMINAL_CANDIDATE",
            "option_outcomes_opened": 0,
        }
        try:
            obs = env.reset_fresh(reset_seed)
            state = env.flat_state()
            record["source_state_sha256"] = array_sha256(state)
            record["scene_fingerprint"] = scene_fingerprint(obs)
            record["source_state_dtype"] = str(state.dtype)
            record["source_state"] = state.tolist()
            policy.reset()
            obs = env.reset_to_exact(state, model_xml=env.model_xml())
            for _ in range(10):
                obs, _, _, _ = env.step(env.dummy_action())
            success = False
            termination = "max_policy_steps"
            action_hashes = []
            for _ in range(args.max_policy_steps):
                policy_obs = env.policy_observation(obs, policy.resize_size)
                action = np.asarray(policy.act(policy_obs, env.task_description))
                action_hashes.append(array_sha256(action))
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
                    "action_sha256": action_hashes,
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
                    "action_sha256": [],
                }
            )
        rows.append(record)
    complete = sum(row["status"] == "NOMINAL_COMPLETE" for row in rows)
    successes = sum(bool(row["task_success"]) for row in rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_formal_nominal_shard",
        "task_id": task_id,
        "shard_index": args.shard_index,
        "candidate_index_start": start,
        "candidate_index_end_exclusive": end,
        "source_plan": str(args.source_plan),
        "execution": {
            "git_commit": resolve_git_head(Path.cwd()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "attempted_sources": len(rows),
        "complete_sources": complete,
        "nominal_successes": successes,
        "option_outcomes_opened": 0,
        "rows": rows,
    }
    payload["shard_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "complete": complete, "successes": successes}, sort_keys=True))
    if complete != 21:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
