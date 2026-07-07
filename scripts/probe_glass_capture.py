#!/usr/bin/env python
"""Self-report probe — capture OpenVLA hidden states on Category 2 (glass), three arms.

R4 fit the "I will crash" probe on the static WALL (env_collision). This extends the same
rigorous self-report test to the fragile-object hazard (object_collision, §6d) so the probe
claim is not wall-specific. Data collection only; `scripts/probe_glass_analysis.py` trains the
within-glass probe AND tests cross-hazard transfer of the frozen wall probe.

For the saved glass scenarios (scenarios_glass/) we run the FROZEN policy closed-loop under
three matched conditions — same init state, same task, only the glass changes:

    glass   : glass ON the reach path (treatment)  -> struck, crash by contact_force>=25N
    offpath : glass pushed OFF the path (control)   -> glass VISIBLE but not hit  [confound]
    noglass : glass dropped (movable_objects=None)  -> the normal reach           [negative]

Per step we log the LM's last hidden state h_t + action + robot-vs-glass contact force + eef
pos + whether/when the episode ultimately crashes. Output (mirrors results/selfreport/ schema):
    results/selfreport_glass/hidden.npz   (H: [N, hidden] float16)
    results/selfreport_glass/meta.json    (per-row metadata, aligned to H)

Run on a GPU node:  cd ~/crash_bench/setup && sbatch probe_glass.sbatch
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

OUT = "results/selfreport_glass"
os.makedirs(OUT, exist_ok=True)
SETTLE = 10
GLASS_BODY = "glass_1"
K_REP = 3                      # OpenVLA is nondeterministic; repeat to get enough crashed frames


def glass_force(sim):
    return sim.max_contact_force(list(ROBOT_CONTACT_BODIES), against=[GLASS_BODY])


def rollout(env, policy, sc, drop_glass: bool):
    """One closed-loop episode with per-step hidden capture. drop_glass=True runs the nominal
    reach (no injected cup). Returns list of step-dicts + the crash step (or -1)."""
    movable = None if drop_glass else (sc.movable_objects or None)
    gx = (sc.movable_objects[0]["pos"][0] if (sc.movable_objects and not drop_glass) else float("nan"))
    obs = env.reset_to(sc.init_state, movable_objects=movable)
    sim = env.sim_view
    # the glass crash predicates reference glass_1; with no glass there is nothing to hit
    crash_pred = None if drop_glass else build_any(sc.crash_predicates)

    for _ in range(SETTLE):                                   # settle (mirror run_episode)
        obs, _, _, _ = env.step(env.dummy_action())

    steps, crash_step = [], -1
    for t in range(sc.max_steps):
        observation = env.policy_observation(obs, policy.resize_size)
        action = policy.act(observation, sc.instruction)
        h = policy.last_hidden
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        gf = 0.0 if drop_glass else glass_force(sim)
        steps.append({
            "t": t,
            "hidden": None if h is None else h.astype(np.float16),
            "act_xyz_norm": float(np.linalg.norm(np.asarray(action[:3], dtype=np.float32))),
            "gripper": float(action[6]),
            "glass_force": float(gf),
            "eef_x": float(eef[0]), "eef_y": float(eef[1]), "eef_z": float(eef[2]),
            "glass_x": float(gx),
        })
        obs, _, _, _ = env.step(action.tolist())
        if crash_pred is not None and crash_pred(sim):
            crash_step = t
            break
    return steps, crash_step


def main():
    # (cond, regime-filter, drop_glass): which saved scenarios, and whether to keep the cup
    conditions = [
        ("glass",   "treatment", False),   # cup on the path
        ("offpath", "control",   False),   # cup off the path (confound)
        ("noglass", "treatment", True),    # cup dropped -> nominal reach (negative)
    ]
    all_scn = [Scenario.load(os.path.dirname(p))
               for p in sorted(glob.glob("scenarios_glass/*/scenario.json"))]
    by_regime = {"treatment": [], "control": []}
    for sc in all_scn:
        by_regime.get(sc.metadata.get("regime", "?"), []).append(sc)

    policy = OpenVLAPolicy(
        pretrained_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
        unnorm_key="libero_spatial", center_crop=True, capture_hidden=True,
    )
    env = LiberoEnv("libero_spatial", 0)

    H, meta = [], []
    for cond, regime, drop in conditions:
        scns = by_regime.get(regime, [])
        print(f"\n=== condition '{cond}' : {len(scns)} scenarios x {K_REP} reps (drop_glass={drop}) ===")
        for sc in scns:
            for rep in range(K_REP):
                steps, cstep = rollout(env, policy, sc, drop_glass=drop)
                crashed = cstep >= 0
                for s in steps:
                    if s["hidden"] is None:
                        continue
                    H.append(s["hidden"])
                    meta.append({
                        "cond": cond, "scenario_id": sc.id, "rep": rep, "t": s["t"],
                        "regime": sc.metadata.get("regime", "?"),
                        "tag": sc.metadata.get("tag", "?"),
                        "clearance": sc.metadata.get("clearance_to_path"),
                        "crashed_episode": bool(crashed),
                        "crash_step": int(cstep),
                        "steps_to_crash": int(cstep - s["t"]) if crashed else -1,
                        "act_xyz_norm": s["act_xyz_norm"], "gripper": s["gripper"],
                        "glass_force": s["glass_force"],
                        "eef_x": s["eef_x"], "eef_y": s["eef_y"], "eef_z": s["eef_z"],
                        "glass_x": s["glass_x"],
                    })
                print(f"  {sc.id:48s} rep{rep} crash={'Y' if crashed else 'n'} "
                      f"crash_step={cstep:3d} steps={len(steps):3d}")

    Harr = np.asarray(H, dtype=np.float16)
    np.savez_compressed(f"{OUT}/hidden.npz", H=Harr)
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    print(f"\nwrote {OUT}/hidden.npz  H={Harr.shape}  and {OUT}/meta.json  rows={len(meta)}")
    tally = {}
    for m in meta:
        tally[m["cond"]] = tally.get(m["cond"], 0) + 1
    print("frames by condition:", tally)


if __name__ == "__main__":
    main()
