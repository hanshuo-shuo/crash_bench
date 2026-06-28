#!/usr/bin/env python
"""Phase 2-④: run OpenVLA closed-loop on the NO-WALL (in-distribution) scenarios and report
crash rate by condition (closed=treatment vs open=control).

This validates the key assumption: a VLA with no pre-crash policy drives into a FAMILIAR,
in-distribution obstacle (a closed microwave door / cabinet drawer) when it lies on the place
path — i.e. the wall-injection 100% crash is not an OOD artifact. The open vs closed pair is a
tight within-scene control (same scene/objects/checkpoint, only the articulation differs).

K_REPEAT rollouts per scenario (OpenVLA is nondeterministic). Saves a video per rollout so the
crash can be eyeballed. Writes results/nowall.json (rows carry condition + rep).

Run on a GPU node (MUJOCO_GL=egl). Uses the libero-10 checkpoint.
"""

from __future__ import annotations

import os, json
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from crashbench.scenario import load_all
from crashbench.envs import LiberoEnv
from crashbench.eval import run_episode

CKPT = "openvla/openvla-7b-finetuned-libero-10"
UNNORM = "libero_10"
SCN = "scenarios_nowall"
K_REPEAT = 5
OUT = "results/nowall.json"
VID = "results/nowall_videos"


def main():
    from crashbench.policies import OpenVLAPolicy
    scenarios = load_all(SCN)
    if not scenarios:
        raise SystemExit(f"no scenarios under {SCN}/ — run scripts/phase2_build_nowall.py first")
    print(f"loaded {len(scenarios)} no-wall scenarios")

    policy = OpenVLAPolicy(pretrained_checkpoint=CKPT, unnorm_key=UNNORM)
    os.makedirs(VID, exist_ok=True)

    env_cache: dict[int, LiberoEnv] = {}
    rows = []
    by_cond = defaultdict(lambda: [0, 0])  # condition -> [crashes, trials]
    for sc in scenarios:
        cond = sc.metadata.get("condition", "?")
        if sc.task_id not in env_cache:
            env_cache[sc.task_id] = LiberoEnv(sc.task_suite, sc.task_id)
        env = env_cache[sc.task_id]
        ncr = 0
        for rep in range(K_REPEAT):
            vid = f"{VID}/{sc.id}__rep{rep}.mp4"
            res = run_episode(sc, env, policy, save_video_path=vid)
            rows.append(res.__dict__ | {"outcome": res.outcome.value, "rep": rep,
                                        "condition": cond, "role": sc.metadata.get("role")})
            ncr += int(res.crashed)
            by_cond[cond][0] += int(res.crashed)
            by_cond[cond][1] += 1
        print(f"  {sc.id:46s} [{cond:6s}] crashed {ncr}/{K_REPEAT} "
              f"peakF~{np.mean([r['peak_contact_force'] for r in rows[-K_REPEAT:]]):.0f}N")

    print("\n=== crash rate by condition ===")
    for cond, (c, n) in sorted(by_cond.items()):
        print(f"  {cond:8s}: {c}/{n} = {c/n:.0%}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT}  (+ videos in {VID}/)")


if __name__ == "__main__":
    main()
