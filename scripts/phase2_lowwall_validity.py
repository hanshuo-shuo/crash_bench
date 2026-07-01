#!/usr/bin/env python
"""Phase 0b validity check: does OpenVLA STILL crash into the LOWERED wall?

The task-completion witness only exists when the wall is short enough that the elbow clears
during the grasp descent (h=0.10, base-preserving -> top z=1.06, from the original 0.22/top=1.30).
Lowering the wall is only legitimate if OpenVLA STILL crashes into it (README: the benchmark
scenario must stay valid — a real hazard the policy walks into). We keep the crash predicate
unchanged (75 N wall contact) and just shrink the wall geometry, then re-run OpenVLA.

Prints, per scenario, the OpenVLA outcome at the ORIGINAL wall (sanity: expect CRASH) and at each
candidate LOWERED height (expect CRASH to persist). Runs on a GPU node (openvla env + GPU).

  python scripts/phase2_lowwall_validity.py --scenarios 'scenarios/*d62*' --heights 0.10 0.12 0.14
"""

from __future__ import annotations

import os, glob, argparse, copy
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.scenario import Scenario
from crashbench.envs import LiberoEnv
from crashbench.policies import OpenVLAPolicy
from crashbench.eval import run_episode


def lower_wall(wall, h):
    """Base-preserving short wall: keep the bottom on the table, drop the top."""
    w = copy.deepcopy(wall); w["size"] = list(w["size"]); w["pos"] = list(w["pos"])
    bottom = wall["pos"][2] - wall["size"][2]
    w["size"][2] = h; w["pos"][2] = bottom + h
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="scenarios/env_collision__T5__libero_spatial_t0_wall_d62")
    ap.add_argument("--heights", type=float, nargs="+", default=[0.10])
    ap.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--unnorm_key", default="libero_spatial")
    args = ap.parse_args()

    scns = [Scenario.load(os.path.dirname(p))
            for p in sorted(glob.glob(os.path.join(args.scenarios, "scenario.json")))]
    print(f"validity check on {len(scns)} scenario(s); heights {args.heights}\n")
    policy = OpenVLAPolicy(pretrained_checkpoint=args.checkpoint, unnorm_key=args.unnorm_key,
                           center_crop=True)

    for sc in scns:
        env = LiberoEnv(sc.task_suite, sc.task_id)
        orig = copy.deepcopy(sc.obstacles[0])
        print(f"  {sc.id}")
        # baseline: original tall wall
        sc.obstacles = [orig]
        r = run_episode(sc, env, policy)
        print(f"    original  h={orig['size'][2]:.2f} top={orig['pos'][2]+orig['size'][2]:.2f} "
              f"-> {r.outcome.value:12s} steps={r.steps_to_event} peakF={r.peak_contact_force:.0f}")
        # candidates: lowered walls
        for h in args.heights:
            w = lower_wall(orig, h)
            sc.obstacles = [w]
            r = run_episode(sc, env, policy)
            valid = "OK (still crashes)" if r.outcome.value == "crash" else "!! NO CRASH — INVALID"
            print(f"    lowered   h={h:.2f} top={w['pos'][2]+w['size'][2]:.2f} "
                  f"-> {r.outcome.value:12s} steps={r.steps_to_event} peakF={r.peak_contact_force:.0f}  {valid}")
        print()


if __name__ == "__main__":
    main()
