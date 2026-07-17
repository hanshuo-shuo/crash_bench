#!/usr/bin/env python
"""Phase 1 pilot entry point (PLAN.md §4 Phase 1, §6 step 5).

Load all scenarios, run a policy closed-loop on each, print the crash rate + the
four metrics. THIS PRODUCES THE GO/NO-GO NUMBER (README §11):
    <20% crash -> no paper | 20-50% -> standard | >50% -> easy paper.

Run on a GPU node (needs the openvla env + a GPU):
    sbatch setup/run_pilot.sbatch          # (wrap this script)
or interactively:
    MUJOCO_GL=egl python scripts/run_pilot.py \
        --scenarios scenarios --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --unnorm_key libero_spatial --out results/pilot.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crashbench.scenario import load_all
from crashbench.envs import LiberoEnv
from crashbench.policies import build_policy, available_backends, canonical_name
from crashbench.eval import run_episode
from crashbench import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="scenarios", help="dir of scenario subdirs")
    ap.add_argument("--policy", default="openvla",
                    help=f"VLA backend (Path 3 cross-policy). one of {available_backends()}")
    ap.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--unnorm_key", default="libero_spatial")
    ap.add_argument("--center_crop", action="store_true", default=True)
    ap.add_argument("--prompt_prefix", default="", help="README §7 prompted-careful baseline")
    ap.add_argument("--out", default="results/pilot.json")
    ap.add_argument("--video_dir", default=None, help="optional dir to dump rollout MP4s")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing result file (unsafe for frozen outputs)")
    args = ap.parse_args()

    scenarios = load_all(args.scenarios)
    if not scenarios:
        raise SystemExit(f"no scenarios found under {args.scenarios}/ — author some first "
                         f"(scripts/author_scenario.py)")
    print(f"loaded {len(scenarios)} scenarios")

    # load policy once (expensive); reuse across scenarios. Same on/off-path protocol,
    # only the backend changes -> Path 3 architecture-independence claim.
    print(f"policy backend: {args.policy}")
    policy = build_policy(
        args.policy,
        pretrained_checkpoint=args.checkpoint,
        unnorm_key=args.unnorm_key,
        center_crop=args.center_crop,
        prompt_prefix=args.prompt_prefix,
    )

    # π0 (openpi/JAX) needs the torch-free env path (no OpenVLA repo on sys.path); every other
    # backend keeps the OpenVLA "experiments.robot" path. Only the env builder differs — scenarios
    # and init_states are identical, so the on/off-path protocol stays apples-to-apples.
    env_family = "pi0" if canonical_name(args.policy) == "pi0" else "openvla"

    results = []
    for sc in scenarios:
        env = LiberoEnv(sc.task_suite, sc.task_id, model_family=env_family)   # TODO(perf): cache env per (suite,task_id)
        video = f"{args.video_dir}/{sc.id}.mp4" if args.video_dir else None
        res = run_episode(sc, env, policy, save_video_path=video)
        results.append(res)
        print(f"  {sc.id:45s} {res.outcome.value:16s} "
              f"steps={res.steps_to_event:3d} peakF={res.peak_contact_force:6.1f}")

    print("\n" + metrics.report(results))

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out}; choose a new --out or pass --overwrite after review")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.__dict__ | {"outcome": r.outcome.value} for r in results], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
