#!/usr/bin/env python
"""Repeated closed-loop pilot runner for small baseline studies.

This is intentionally separate from ``run_pilot.py``: the latter preserves the original
one-pass result format, while this runner records explicit rollout repetitions and a compact
summary. It is used by the prompted-careful baseline so that K is visible in the output rather
than hidden in a shell loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crashbench.envs import LiberoEnv
from crashbench.eval import run_episode
from crashbench.policies import build_policy, canonical_name
from crashbench.scenario import load_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="scenarios")
    ap.add_argument("--policy", default="openvla")
    ap.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--unnorm_key", default="libero_spatial")
    ap.add_argument("--prompt_prefix", default="")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/repeated_pilot.json")
    ap.add_argument("--video_dir", default=None)
    args = ap.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    scenarios = load_all(args.scenarios)
    if not scenarios:
        raise SystemExit(f"no scenarios found under {args.scenarios}/")

    policy = build_policy(
        args.policy,
        pretrained_checkpoint=args.checkpoint,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        prompt_prefix=args.prompt_prefix,
    )
    env_family = "pi0" if canonical_name(args.policy) == "pi0" else "openvla"
    rows = []

    print(f"loaded {len(scenarios)} scenarios; repeats={args.repeats}; policy={args.policy}")
    for rep in range(args.repeats):
        for sc in scenarios:
            env = LiberoEnv(sc.task_suite, sc.task_id, model_family=env_family)
            video = None
            if args.video_dir:
                video = f"{args.video_dir}/rep{rep}/{sc.id}.mp4"
            result = run_episode(sc, env, policy, save_video_path=video)
            row = result.__dict__ | {
                "outcome": result.outcome.value,
                "rep": rep,
                "prompt_prefix": args.prompt_prefix,
            }
            rows.append(row)
            print(f"  rep={rep} {sc.id:48s} {result.outcome.value:16s} "
                  f"steps={result.steps_to_event:3d} peakF={result.peak_contact_force:6.1f}")

    n = len(rows)
    crashes = sum(bool(x["crashed"]) for x in rows)
    successes = sum(bool(x["succeeded"]) for x in rows)
    summary = {
        "n": n,
        "crashed": crashes,
        "crash_rate": crashes / n if n else None,
        "succeeded": successes,
        "success_rate": successes / n if n else None,
    }
    out = {
        "config": {
            "scenarios": args.scenarios,
            "policy": args.policy,
            "checkpoint": args.checkpoint,
            "unnorm_key": args.unnorm_key,
            "repeats": args.repeats,
            "prompt_prefix": args.prompt_prefix,
        },
        "summary": summary,
        "episodes": rows,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
