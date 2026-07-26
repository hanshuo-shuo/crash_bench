#!/usr/bin/env python
"""Nominal OpenVLA gate for selecting cross-task M1 candidates.

Runs clean LIBERO-Spatial tasks without injected hazards. A task only proceeds to the
cross-task corridor experiment if its nominal success rate is above the configured gate.
The script deliberately uses the CrashBench observation/action bridge, so the gate tests the
same code path as the later hazard evaluation.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.envs import LiberoEnv
from crashbench.policies import OpenVLAPolicy
from crashbench.provenance import (
    repository_provenance, require_checkpoint_revision, runtime_provenance,
    write_json_exclusive,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_one(env, policy, state, max_steps: int, settle: int) -> dict:
    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset_to(state)
    for _ in range(settle):
        obs, _, done, _ = env.step(env.dummy_action())
        if done:
            return {"success": True, "steps": 0, "peak_force": float(env.sim_view.peak_force)}

    for step in range(max_steps):
        observation = env.policy_observation(obs, policy.resize_size)
        action = policy.act(observation, env.task_description)
        obs, _, done, _ = env.step(action.tolist() if hasattr(action, "tolist") else action)
        if done:
            return {"success": True, "steps": step + 1,
                    "peak_force": float(env.sim_view.peak_force)}
    return {"success": False, "steps": max_steps,
            "peak_force": float(env.sim_view.peak_force)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_ids", default="1,2,3,4,5,6,7,8,9",
                    help="comma-separated task ids; task 0 is already the anchor task")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=220)
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--checkpoint-revision", default=None)
    ap.add_argument("--unnorm_key", default="libero_spatial")
    ap.add_argument("--out", default="results/m1_nominal_gate.json")
    ap.add_argument("--base-seed", type=int, default=20260726)
    ap.add_argument("--fail_below", type=float, default=None,
                    help="exit nonzero if any task is below this success rate")
    args = ap.parse_args()
    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    if not task_ids or args.repeats < 1:
        raise SystemExit("need at least one task and --repeats >= 1")

    revision = require_checkpoint_revision(args.checkpoint_revision) if args.checkpoint_revision else None
    repo = repository_provenance(ROOT, require_clean=revision is not None)
    seed_everything(args.base_seed)
    policy = OpenVLAPolicy(
        pretrained_checkpoint=args.checkpoint,
        checkpoint_revision=revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
    )
    rows = []
    print(f"nominal gate: suite={args.suite} tasks={task_ids} repeats={args.repeats}")
    for task_id in task_ids:
        env = LiberoEnv(args.suite, task_id)
        states = np.asarray(env.default_init_states())
        task_rows = []
        for rep in range(args.repeats):
            episode_seed = args.base_seed + task_id * 1000 + rep
            seed_everything(episode_seed)
            env.seed(episode_seed)
            state = states[rep % len(states)]
            result = run_one(env, policy, state, args.max_steps, args.settle)
            row = {
                "suite": args.suite,
                "task_id": task_id,
                "task_description": env.task_description,
                "rep": rep,
                "episode_seed": episode_seed,
                **result,
            }
            rows.append(row)
            task_rows.append(row)
            print(f"  task={task_id:2d} rep={rep} success={result['success']} "
                  f"steps={result['steps']:3d}")
        rate = sum(x["success"] for x in task_rows) / len(task_rows)
        print(f"  => task {task_id}: {sum(x['success'] for x in task_rows)}/"
              f"{len(task_rows)} = {rate:.1%}")

    by_task = {}
    for task_id in task_ids:
        task_rows = [x for x in rows if x["task_id"] == task_id]
        successes = sum(x["success"] for x in task_rows)
        by_task[str(task_id)] = {
            "task_description": task_rows[0]["task_description"],
            "n": len(task_rows),
            "successes": successes,
            "success_rate": successes / len(task_rows),
        }
    out = {
        "schema_version": 1,
        "config": vars(args),
        "repository": repo,
        "runtime": runtime_provenance(),
        "checkpoint": policy.checkpoint_identity,
        "tasks": by_task,
        "episodes": rows,
    }
    path = Path(args.out)
    write_json_exclusive(path, out)
    print(f"wrote {path}")

    if args.fail_below is not None:
        bad = [tid for tid, x in by_task.items() if x["success_rate"] < args.fail_below]
        if bad:
            raise SystemExit(f"nominal gate failed for task(s): {', '.join(bad)}")


if __name__ == "__main__":
    main()
