#!/usr/bin/env python
"""Self-report probe capture — MODEL-AGNOSTIC (Path 3 cross-architecture probe fit).

Generalizes scripts/probe_selfreport.py to any registered backend (openvla / openvla-oft / pi0)
so the "crash is linearly decodable from the model's own hidden state" test runs on all three
architectures with ONE code path. Same protocol, same scenarios; only `--policy` / `--out` change.

For each of the 5 env_collision treatment scenarios we run the FROZEN policy closed-loop under
matched conditions (the only change is the wall):

    nowall : obstacles dropped   -> the normal reach (mostly no crash)
    wall   : wall on the path    -> crash 100%
    offpath: control walls        -> wall VISIBLE but off the path (no crash)   [confound control]

Chunked backends (OFT: 8-step open-loop; π0: 5-step replan) run a real forward ONLY on the
requery step; on buffered steps the wrapper sets last_hidden=None and we skip the frame — so
every logged row is a genuine hidden state, at its true timestep. Output (schema identical to
probe_selfreport.py, consumed by probe_selfreport_analysis.py IN=<out dir>):
    <out>/hidden.npz   (H: [N, hidden] float16)
    <out>/meta.json    (per-row metadata, aligned to H)

GPU node:  sbatch setup/probe_selfreport_oft.sbatch   |   sbatch setup/probe_selfreport_pi0.sbatch
"""

from __future__ import annotations

import argparse
import os, json, glob
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.scenario import Scenario
from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES
from crashbench.policies import build_policy, canonical_name
from crashbench.predicates import build_any

SETTLE = 10


def wall_force(sim):
    return sim.max_contact_force(list(ROBOT_CONTACT_BODIES), against=["crash_wall"])


def rollout(env, policy, sc, drop_wall: bool):
    """One closed-loop episode with per-step capture. Returns (step-dicts, crash_step).
    A step's 'hidden' is None whenever the policy served the action from its open-loop buffer
    (no fresh forward) — those rows are dropped downstream."""
    obstacles = None if drop_wall else (sc.obstacles or None)
    wall = (sc.obstacles[0] if (sc.obstacles and not drop_wall) else None)
    obs = env.reset_to(sc.init_state, obstacles=obstacles)
    sim = env.sim_view
    crash_pred = build_any(sc.crash_predicates)
    if hasattr(policy, "reset"):
        policy.reset()                                            # clear chunk buffer per episode

    for _ in range(SETTLE):                                       # settle (mirror run_episode)
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
            "hidden": None if h is None else np.asarray(h).astype(np.float16),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="openvla",
                    help="backend name (openvla / openvla-oft / pi0)")
    ap.add_argument("--out", required=True, help="output dir, e.g. results/selfreport_oft")
    ap.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--unnorm_key", default="libero_spatial")
    ap.add_argument("--center_crop", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env_family = "pi0" if canonical_name(args.policy) == "pi0" else "openvla"

    # (label, glob, drop_wall)
    conditions = [
        ("nowall",  "scenarios/*/scenario.json",         True),
        ("wall",    "scenarios/*/scenario.json",         False),
        ("offpath", "scenarios_control/*/scenario.json", False),
    ]

    print(f"policy backend: {args.policy}  (env_family={env_family})", flush=True)
    policy = build_policy(
        args.policy,
        pretrained_checkpoint=args.checkpoint,
        unnorm_key=args.unnorm_key,
        center_crop=args.center_crop,
        capture_hidden=True,
    )

    # single env reused across scenarios (all treatment scenarios share suite/task); take
    # suite/task from the first wall scenario so this is not hard-coded to libero_spatial t0.
    first = Scenario.load(os.path.dirname(sorted(glob.glob("scenarios/*/scenario.json"))[0]))
    env = LiberoEnv(first.task_suite, first.task_id, model_family=env_family)

    def flush(H, meta, note):
        # incremental save after each condition: a walltime kill still leaves usable data
        # (wall+nowall is the core probe fit; offpath is the slow, less-essential confound).
        Harr = np.asarray(H, dtype=np.float16)
        np.savez_compressed(f"{args.out}/hidden.npz", H=Harr)
        json.dump(meta, open(f"{args.out}/meta.json", "w"))
        from collections import Counter
        c = Counter((m["cond"], m["crashed_episode"]) for m in meta)
        print(f"  [flush after {note}] H={Harr.shape} rows={len(meta)}  {dict(c)}", flush=True)

    H, meta = [], []
    for cond, pat, drop in conditions:
        scns = [Scenario.load(os.path.dirname(p)) for p in sorted(glob.glob(pat))]
        print(f"\n=== condition '{cond}' : {len(scns)} scenarios (drop_wall={drop}) ===", flush=True)
        for sc in scns:
            steps, cstep = rollout(env, policy, sc, drop_wall=drop)
            crashed = cstep >= 0
            n_hidden = 0
            for s in steps:
                if s["hidden"] is None:
                    continue
                n_hidden += 1
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
                  f"crash_step={cstep:3d} steps={len(steps):3d} hidden_frames={n_hidden:3d}", flush=True)
        flush(H, meta, f"condition '{cond}'")

    Harr = np.asarray(H, dtype=np.float16)
    print(f"\nwrote {args.out}/hidden.npz  H={Harr.shape}  and {args.out}/meta.json  rows={len(meta)}")
    from collections import Counter
    c = Counter((m["cond"], m["crashed_episode"]) for m in meta)
    for k, v in sorted(c.items()):
        print(f"   {k}: {v} frames")


if __name__ == "__main__":
    main()
