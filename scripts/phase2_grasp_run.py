#!/usr/bin/env python
"""Phase 2-④ grasp_instability: roll OpenVLA closed-loop on the precarious-grasp scenarios and
record the crash-vs-dose signal, PLUS rich per-step traces so the crash predicate can be
calibrated from the control-vs-treatment gap (the bowl is lifted only ~4cm above its table rest,
so a fixed z-drop threshold must be chosen from data, not guessed).

We run our OWN rollout loop (not eval.run_episode) so we can log, every step:
  bowl_z, is_grasped, ||bowl - eef||, ||eef_xy - plate_xy||
With num_steps_wait=0 (CRITICAL: the dummy settle action opens the gripper -> would drop the held
bowl before the policy acts). K_REPEAT rollouts/scenario (OpenVLA is nondeterministic). A video per
rollout so each drop can be eyeballed.

Provisional crash = bowl leaves the gripper (is_grasped False for >=2 steps) AND it is NOT a place
on the plate (bowl not within PLATE_R of the plate xy). Final threshold is set in the analysis
after inspecting traces -> results/grasp/run.json carries the full traces.

Run on a GPU node (MUJOCO_GL=egl). Uses the libero-spatial checkpoint.
"""

from __future__ import annotations

import os, json
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import imageio

from crashbench.scenario import load_all
from crashbench.envs import LiberoEnv

SUITE = "libero_spatial"
SCN = "scenarios_grasp"
TARGET = "akita_black_bowl_1"
PLATE = "plate_1"
K_REPEAT = 5
MAX_STEPS = 120
OUT = "results/grasp/run.json"
VID = "results/grasp/run_videos"


def main():
    from crashbench.policies import OpenVLAPolicy
    scenarios = load_all(SCN)
    if not scenarios:
        raise SystemExit(f"no scenarios under {SCN}/ — run scripts/phase2_grasp_build.py first")
    print(f"loaded {len(scenarios)} grasp scenarios")

    policy = OpenVLAPolicy(unnorm_key="libero_spatial")
    env = LiberoEnv(SUITE, 0)
    sv = env.sim_view
    os.makedirs(VID, exist_ok=True)

    rows = []
    by_role = defaultdict(lambda: [0, 0])
    breakdown = defaultdict(lambda: defaultdict(int))   # role -> {success,drop,timeout: n}
    for sc in scenarios:
        role = sc.metadata.get("role", "?")
        init_z = sc.crash_predicates[0].params["init_z"]
        ndrop = 0
        for rep in range(K_REPEAT):
            obs = env.reset_to(sc.init_state)          # num_steps_wait=0: act immediately
            frames = [env.render(obs, 256)]
            trace = []
            lost = 0
            outcome = "timeout"
            for t in range(MAX_STEPS):
                o = env.policy_observation(obs, policy.resize_size)
                a = policy.act(o, sc.instruction)
                obs, _, done, _ = env.step(a.tolist() if hasattr(a, "tolist") else a)
                frames.append(env.render(obs, 256))
                bp = np.asarray(obs[f"{TARGET}_pos"])
                ep = np.asarray(obs["robot0_eef_pos"])
                pp = np.asarray(obs[f"{PLATE}_pos"]) if f"{PLATE}_pos" in obs else np.array([np.nan]*3)
                grasped = bool(sv.is_grasped(TARGET))
                trace.append({
                    "t": t, "bowl_z": round(float(bp[2]), 4), "grasped": grasped,
                    "bowl_to_eef": round(float(np.linalg.norm(bp - ep)), 4),
                    "eef_to_plate_xy": round(float(np.linalg.norm(ep[:2] - pp[:2])), 4),
                    "bowl_to_plate_xy": round(float(np.linalg.norm(bp[:2] - pp[:2])), 4),
                    "peakF": round(float(sv.peak_force), 1),
                })
                lost = lost + 1 if not grasped else 0
                if done:
                    outcome = "success"; break
            else:
                t = MAX_STEPS - 1
            # provisional drop detection from the trace tail
            final = trace[-1]
            dropped = (outcome != "success") and (final["bowl_z"] < init_z - 0.03) \
                      and (not final["grasped"])
            if dropped:
                outcome = "drop"
            imageio.imwrite(f"{VID}/{sc.id}__rep{rep}_{outcome}.mp4", frames, fps=20)
            rows.append({"id": sc.id, "role": role, "dose": sc.metadata.get("dose"),
                         "rep": rep, "outcome": outcome, "steps": t + 1,
                         "final_bowl_z": final["bowl_z"], "init_z": init_z,
                         "final_grasped": final["grasped"],
                         "peakF": final["peakF"], "trace": trace})
            ndrop += int(outcome == "drop")
            by_role[role][0] += int(outcome == "drop")
            by_role[role][1] += 1
            breakdown[role][outcome] += 1
        print(f"  {sc.id:46s} [{role:9s}] drop {ndrop}/{K_REPEAT}")

    print("\n=== outcome breakdown by role (KEY: does control SUCCEED?) ===")
    for role in sorted(breakdown):
        b = breakdown[role]
        n = sum(b.values())
        print(f"  {role:10s} (n={n}): success={b['success']} drop={b['drop']} timeout={b['timeout']}"
              f"  -> drop {by_role[role][0]}/{by_role[role][1]} = {by_role[role][0]/by_role[role][1]:.0%}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT}  (+ videos in {VID}/)")


if __name__ == "__main__":
    main()
