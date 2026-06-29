#!/usr/bin/env python
"""Path 1-1b — activation steering (PLAN.md §11 Stage 1 experiment 1b, the mechanism upside).

Result 5 (1a) gated a retreat ON the probe. 1b asks a deeper question: is the probe direction
itself *steerable* — if we push the model's hidden state AWAY from "I will crash", does it brake
on its own, with NO hand-coded controller? We subtract alpha * d_unit (d_unit = the probe's crash
direction) from OpenVLA's final-RMSNorm output at every token position. Because the action is a
discrete token off the LM head applied to that post-norm state, this directly moves the action
bins (PLAN.md §11). alpha=0 reproduces the bare policy.

We sweep alpha and, on the 5 on-path walls, ask: does peak contact force fall monotonically? And
on the 5 no-wall reaches, ask: is there an alpha window that lowers force WITHOUT destroying the
nominal task (success preserved)? That window is the deliverable.

Caveat (PLAN.md §11): the probe was trained on the prefill last-token; steering also acts on the
decode positions, whose subspace may not align — so this is an empirical test, not a guarantee.

Run on a GPU node:  cd setup && sbatch phase3_steering.sbatch
"""

from __future__ import annotations

import os, json, glob
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.scenario import Scenario
from crashbench.envs import LiberoEnv
from crashbench.policies import OpenVLAPolicy
from crashbench.probe import Probe
from crashbench.predicates import build_any, build_predicate

OUT = "results/steering"
os.makedirs(OUT, exist_ok=True)
SETTLE = 10
ALPHAS = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
TREAT_K = 2
NOWALL_K = 1


def rollout(env, policy, sc, drop_wall: bool):
    obstacles = None if drop_wall else (sc.obstacles or None)
    obs = env.reset_to(sc.init_state, obstacles=obstacles)
    sim = env.sim_view
    crash_pred = build_any(sc.crash_predicates)
    success_pred = build_predicate(sc.success_predicate)
    for _ in range(SETTLE):
        obs, _, _, _ = env.step(env.dummy_action())
    act_mags, crashed, succeeded, crash_step = [], False, False, -1
    for t in range(sc.max_steps):
        observation = env.policy_observation(obs, policy.resize_size)
        action = policy.act(observation, sc.instruction)
        act_mags.append(float(np.linalg.norm(np.asarray(action[:3], dtype=np.float32))))
        obs, _, _, _ = env.step(action.tolist())
        if crash_pred(sim):
            crashed, crash_step = True, t
            break
        if success_pred(sim):
            succeeded = True
            break
    return {
        "crashed": bool(crashed), "succeeded": bool(succeeded),
        "peak_force": round(float(sim.peak_force), 1), "crash_step": int(crash_step),
        "mean_act_mag": round(float(np.mean(act_mags)), 4) if act_mags else None,
        "n_steps": len(act_mags),
    }


def main():
    treat = [Scenario.load(os.path.dirname(p))
             for p in sorted(glob.glob("scenarios/*/scenario.json"))]
    print(f"treatment walls: {len(treat)}   alphas: {ALPHAS}")

    probe = Probe.load()
    d_unit = probe.steer_vector()
    print(f"probe loaded; steer_vector |d|={np.linalg.norm(d_unit):.3f}")

    policy = OpenVLAPolicy(
        pretrained_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
        unnorm_key="libero_spatial", center_crop=True, enable_steering=True,
    )
    env = LiberoEnv("libero_spatial", 0)

    rows = []
    for alpha in ALPHAS:
        policy.set_steering(d_unit, alpha)
        print(f"\n=== alpha={alpha} ===")
        for sc in treat:                                    # on-path walls
            for k in range(TREAT_K):
                r = rollout(env, policy, sc, drop_wall=False)
                rows.append({"cond": "treatment", "alpha": alpha, "scenario_id": sc.id, "k": k, **r})
                print(f"  [wall ] {sc.id[-22:]:22s} k{k} crash={r['crashed']} "
                      f"peakF={r['peak_force']:6.1f} succ={r['succeeded']} amag={r['mean_act_mag']}")
        for sc in treat:                                    # no-wall benign reach
            for k in range(NOWALL_K):
                r = rollout(env, policy, sc, drop_wall=True)
                rows.append({"cond": "nowall", "alpha": alpha, "scenario_id": sc.id, "k": k, **r})
                print(f"  [nowal] {sc.id[-22:]:22s} k{k} crash={r['crashed']} "
                      f"peakF={r['peak_force']:6.1f} succ={r['succeeded']} amag={r['mean_act_mag']}")

    json.dump(rows, open(f"{OUT}/sweep.json", "w"), indent=1)

    # ---------- summary per alpha ----------
    def agg(cond, alpha):
        r = [x for x in rows if x["cond"] == cond and x["alpha"] == alpha]
        n = len(r)
        return {
            "n": n,
            "crash_rate": round(sum(x["crashed"] for x in r) / n, 3) if n else None,
            "success_rate": round(sum(x["succeeded"] for x in r) / n, 3) if n else None,
            "peak_force_mean": round(float(np.mean([x["peak_force"] for x in r])), 1) if n else None,
            "act_mag_mean": round(float(np.mean([x["mean_act_mag"] for x in r if x["mean_act_mag"] is not None])), 4) if n else None,
        }

    summary = {"alphas": ALPHAS,
               "treatment": {a: agg("treatment", a) for a in ALPHAS},
               "nowall": {a: agg("nowall", a) for a in ALPHAS}}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/sweep.json and {OUT}/summary.json")


if __name__ == "__main__":
    main()
