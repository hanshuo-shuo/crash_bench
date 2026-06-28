#!/usr/bin/env python
"""Self-report probe — capture OpenVLA's internal state, WALL vs NO-WALL (README §8, §11 pilot 2).

OpenVLA is action-only: you cannot *ask* it "are you about to crash?". So we test self-report
the rigorous way — decode crash-imminence from its own hidden states with a linear probe. This
script does the data collection; `scripts/probe_selfreport_analysis.py` trains the probe.

For each of the 5 env_collision treatment scenarios we run the FROZEN policy closed-loop under
matched conditions (same init state, same task — the only change is the wall):

    nowall : obstacles dropped  -> the normal reach (mostly no crash)
    wall   : wall on the path   -> crash 100%
    offpath: control walls       -> wall VISIBLE but off the path (no crash)  [confound control]

Per step we log: the LM's last hidden state h_t, the action, the wall contact force, the eef
position, and whether/when the episode ultimately crashes. Output:
    results/selfreport/hidden.npz   (H: [N, hidden] float16)
    results/selfreport/meta.json    (per-row metadata, aligned to H)

Run on a GPU node:  sbatch setup/probe_selfreport.sbatch
"""

from __future__ import annotations

import os, json, glob
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.scenario import Scenario
from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES
from crashbench.policies import OpenVLAPolicy
from crashbench.predicates import build_any

OUT = "results/selfreport"
os.makedirs(OUT, exist_ok=True)
SETTLE = 10


def wall_force(sim):
    return sim.max_contact_force(list(ROBOT_CONTACT_BODIES), against=["crash_wall"])


def rollout(env, policy, sc, drop_wall: bool):
    """One closed-loop episode with per-step capture. Returns a list of step-dicts (one per
    policy step) plus the crash step (or -1). Hidden state stored separately as float16."""
    obstacles = None if drop_wall else (sc.obstacles or None)
    wall = (sc.obstacles[0] if (sc.obstacles and not drop_wall) else None)
    obs = env.reset_to(sc.init_state, obstacles=obstacles)
    sim = env.sim_view
    crash_pred = build_any(sc.crash_predicates)

    for _ in range(SETTLE):                                   # settle (mirror run_episode)
        obs, _, _, _ = env.step(env.dummy_action())

    steps, crash_step = [], -1
    for t in range(sc.max_steps):
        observation = env.policy_observation(obs, policy.resize_size)
        action = policy.act(observation, sc.instruction)
        h = policy.last_hidden
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        wf = wall_force(sim)
        steps.append({
            "t": t,
            "hidden": None if h is None else h.astype(np.float16),
            "act_xyz_norm": float(np.linalg.norm(np.asarray(action[:3], dtype=np.float32))),
            "gripper": float(action[6]),
            "wall_force": float(wf),
            "eef_x": float(eef[0]), "eef_y": float(eef[1]), "eef_z": float(eef[2]),
            "wall_x": float(wall["pos"][0]) if wall else float("nan"),
        })
        obs, _, _, _ = env.step(action.tolist())
        if crash_pred(sim):
            crash_step = t
            break
    return steps, crash_step


def main():
    # (label, glob, drop_wall)
    conditions = [
        ("nowall",  "scenarios/*/scenario.json",         True),
        ("wall",    "scenarios/*/scenario.json",         False),
        ("offpath", "scenarios_control/*/scenario.json", False),
    ]
    policy = OpenVLAPolicy(
        pretrained_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
        unnorm_key="libero_spatial", center_crop=True, capture_hidden=True,
    )
    env = LiberoEnv("libero_spatial", 0)

    H, meta = [], []
    for cond, pat, drop in conditions:
        scns = [Scenario.load(os.path.dirname(p)) for p in sorted(glob.glob(pat))]
        print(f"\n=== condition '{cond}' : {len(scns)} scenarios (drop_wall={drop}) ===")
        for sc in scns:
            steps, cstep = rollout(env, policy, sc, drop_wall=drop)
            crashed = cstep >= 0
            for s in steps:
                if s["hidden"] is None:
                    continue
                H.append(s["hidden"])
                meta.append({
                    "cond": cond, "scenario_id": sc.id, "t": s["t"],
                    "crashed_episode": bool(crashed),
                    "crash_step": int(cstep),
                    "steps_to_crash": int(cstep - s["t"]) if crashed else -1,
                    "act_xyz_norm": s["act_xyz_norm"], "gripper": s["gripper"],
                    "wall_force": s["wall_force"],
                    "eef_x": s["eef_x"], "eef_y": s["eef_y"], "eef_z": s["eef_z"],
                    "wall_x": s["wall_x"],
                })
            print(f"  {sc.id:48s} crash={'Y' if crashed else 'n'} "
                  f"crash_step={cstep:3d} steps={len(steps):3d}")

    Harr = np.asarray(H, dtype=np.float16)
    np.savez_compressed(f"{OUT}/hidden.npz", H=Harr)
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    print(f"\nwrote {OUT}/hidden.npz  H={Harr.shape}  and {OUT}/meta.json  rows={len(meta)}")
    # quick condition tally
    from collections import Counter
    c = Counter((m["cond"], m["crashed_episode"]) for m in meta)
    for k, v in sorted(c.items()):
        print(f"   {k}: {v} frames")


if __name__ == "__main__":
    main()
