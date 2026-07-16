#!/usr/bin/env python
"""Path 1-1a — probe-triggered safe-abort (PLAN.md §11 Stage 1, the headline number).

R4 showed OpenVLA's frozen hidden state linearly KNOWS a crash is imminent (AUC ~1.0) yet
the policy does not brake. This script closes the loop: we reuse that exact probe to TRIGGER
the witnessed 0 N retreat the moment it fires, turning a passive detector into a causal
intervention — the one thing SALSA / SAFE / Basu et al. stop short of.

Four conditions, all through the identical model-agnostic `run_episode`:

  treatment_baseline : 5 on-path walls, bare OpenVLA          -> expect crash ~100%, ~250 N
  treatment_guarded  : 5 on-path walls, GuardedPolicy         -> expect crash ~0%,  ~0 N
  nowall_guarded     : same 5 scenes, wall dropped (benign)   -> guard must NOT fire (nominal kept)
  offpath_guarded    : 6 'clear' control walls (clr>0.18)     -> guard must NOT fire (decodes
                                                                 'I will crash', not 'a wall exists')

The guard threshold was fixed offline on the negative pool (off-path + no-wall frames) in
probe_selfreport_analysis.py, so a near-zero false-trigger rate here is by construction, and
serves as the confound control: the guard acts on crash-imminence, not wall presence.

Run on a GPU node:  cd setup && sbatch phase3_intervention.sbatch
"""

from __future__ import annotations

import os, json, glob, copy
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.scenario import Scenario
from crashbench.envs import LiberoEnv
from crashbench.policies import OpenVLAPolicy, GuardedPolicy
from crashbench.probe import Probe
from crashbench.eval import run_episode, Outcome

OUT = os.environ.get("CB_OUT", "results/intervention")
os.makedirs(OUT, exist_ok=True)
TREAT_K = int(os.environ.get("CB_TREAT_K", "3"))
FP_K = int(os.environ.get("CB_FP_K", "2"))
TREAT_SCEN = os.environ.get("CB_TREAT_SCEN", "scenarios")
OFFPATH_GLOB = os.environ.get("CB_OFFPATH_GLOB", "scenarios_control/*_clear/scenario.json")


def drop_obstacles(sc: Scenario) -> Scenario:
    sc2 = copy.deepcopy(sc)
    sc2.obstacles = None
    return sc2


def episode_record(cond, sc, k, res, guard=None):
    rec = {
        "cond": cond, "scenario_id": sc.id, "k": k,
        "outcome": res.outcome.value, "crashed": bool(res.crashed),
        "succeeded": bool(res.succeeded),
        "peak_contact_force": round(float(res.peak_contact_force), 1),
        "steps_to_event": int(res.steps_to_event),
    }
    if guard is not None:
        rec["guard_fired"] = bool(guard.aborting)
        rec["trigger_step"] = int(guard.trigger_step)
        rec["logits"] = [None if (l != l) else round(float(l), 3) for l in guard.logits]
    return rec


def main():
    treat = [Scenario.load(os.path.dirname(p))
             for p in sorted(glob.glob(f"{TREAT_SCEN}/*/scenario.json"))]
    offpath = [Scenario.load(os.path.dirname(p))
               for p in sorted(glob.glob(OFFPATH_GLOB))]
    print(f"treatment walls: {len(treat)}   off-path 'clear' walls: {len(offpath)}")

    probe = Probe.load()
    print(f"probe loaded, thr={probe.thr:.3f}")

    base = OpenVLAPolicy(
        pretrained_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
        unnorm_key="libero_spatial", center_crop=True, capture_hidden=True,
    )
    guard = GuardedPolicy(base, probe)
    env = LiberoEnv("libero_spatial", 0)

    rows = []

    # (1) treatment baseline — bare OpenVLA on on-path walls
    print("\n=== treatment_baseline (bare OpenVLA) ===")
    for sc in treat:
        for k in range(TREAT_K):
            res = run_episode(sc, env, base)
            rows.append(episode_record("treatment_baseline", sc, k, res))
            print(f"  {sc.id:42s} k{k} {res.outcome.value:16s} peakF={res.peak_contact_force:6.0f}N")

    # (2) treatment guarded — probe-triggered safe-abort
    print("\n=== treatment_guarded (probe -> retreat) ===")
    for sc in treat:
        for k in range(TREAT_K):
            guard.reset()
            res = run_episode(sc, env, guard)
            rows.append(episode_record("treatment_guarded", sc, k, res, guard))
            print(f"  {sc.id:42s} k{k} {res.outcome.value:16s} peakF={res.peak_contact_force:6.0f}N "
                  f"fired={guard.aborting} @step={guard.trigger_step}")

    # (3) no-wall guarded — benign reach, guard must NOT fire
    print("\n=== nowall_guarded (benign reach, false-trigger check) ===")
    for sc in treat:
        nw = drop_obstacles(sc)
        for k in range(FP_K):
            guard.reset()
            res = run_episode(nw, env, guard)
            rows.append(episode_record("nowall_guarded", nw, k, res, guard))
            print(f"  {nw.id:42s} k{k} {res.outcome.value:16s} fired={guard.aborting} "
                  f"@step={guard.trigger_step}")

    # (4) off-path guarded — wall visible but off path, guard must NOT fire
    print("\n=== offpath_guarded (visible wall off-path, confound) ===")
    for sc in offpath:
        for k in range(FP_K):
            guard.reset()
            res = run_episode(sc, env, guard)
            rows.append(episode_record("offpath_guarded", sc, k, res, guard))
            print(f"  {sc.id:42s} k{k} {res.outcome.value:16s} peakF={res.peak_contact_force:6.0f}N "
                  f"fired={guard.aborting} @step={guard.trigger_step}")

    json.dump(rows, open(f"{OUT}/episodes.json", "w"), indent=1)

    # ---------- summary ----------
    def agg(cond):
        r = [x for x in rows if x["cond"] == cond]
        n = len(r)
        crashed = [x for x in r if x["crashed"]]
        fired = sum(x.get("guard_fired", False) for x in r)
        return {
            "n": n,
            "crash_rate": round(len(crashed) / n, 3) if n else None,
            "n_crash": len(crashed),
            "peak_force_mean_all": round(float(np.mean([x["peak_contact_force"] for x in r])), 1) if n else None,
            "peak_force_mean_crashed": round(float(np.mean([x["peak_contact_force"] for x in crashed])), 1) if crashed else 0.0,
            "guard_fire_rate": round(fired / n, 3) if n else None,
            "n_guard_fired": int(fired),
        }

    summary = {c: agg(c) for c in
               ["treatment_baseline", "treatment_guarded", "nowall_guarded", "offpath_guarded"]}
    summary["headline"] = {
        "crash_rate_baseline_to_guarded": [summary["treatment_baseline"]["crash_rate"],
                                           summary["treatment_guarded"]["crash_rate"]],
        "peak_force_baseline_to_guarded": [summary["treatment_baseline"]["peak_force_mean_crashed"],
                                           summary["treatment_guarded"]["peak_force_mean_all"]],
        "false_trigger_rate_nowall": summary["nowall_guarded"]["guard_fire_rate"],
        "false_trigger_rate_offpath": summary["offpath_guarded"]["guard_fire_rate"],
        "thr": round(probe.thr, 3),
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/episodes.json and {OUT}/summary.json")


if __name__ == "__main__":
    main()
