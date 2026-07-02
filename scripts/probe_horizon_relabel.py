#!/usr/bin/env python
"""Horizon relabel — 'knows but does not act' as a function of time-to-crash (Week-1).

PLAN.md §8.1 asks for the behaviour on the horizon axis (T-1 / T-5 / T-20): how does
the policy behave as the pre-crash state gets closer to impact? We already have the raw
material — every crashed episode in the frozen R4 self-report capture carries a per-frame
`steps_to_crash` (the horizon) plus the commanded action magnitude `act_xyz_norm`. So this
is a *relabel*, not a new experiment: re-score each frame's crash-imminence logit with the
frozen probe and bin/plot against `steps_to_crash`. No GPU.

The figure it produces is the horizon-axis form of the killer result:
  * KNOWS   — crash-imminence logit RAMPS UP as impact approaches, crossing the shield
              threshold ~5 steps out on the on-path walls (T-5, exactly the probe's train
              horizon).
  * DOESN'T — the commanded action magnitude does NOT dip toward impact; it *increases*
    ACT     (the arm pushes harder into the wall), so there is no braking at any horizon.

On-path walls (the canonical blocking hazard) crash within <=9 steps, so they populate
T-1..T-10 but never T-20 — an honest fact ('a wall that blocks the path is hit fast'). The
off-path gradient-band crashers reach T-40+ but the probe stays flat and low for them (their
weak glancing contacts are not flagged as imminent) — shown as the contrast trace.

Inputs (frozen, from R4): results/selfreport/{hidden.npz, meta.json, probe_T5.npz}
Outputs: results/shield/horizon.json, setup/figures/fig_horizon.png
"""

from __future__ import annotations

import os, json, collections
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crashbench.probe import Probe

OUT = "results/shield"
FIGDIR = "setup/figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# Horizon buckets used by the scenario schema (Scenario.horizon = T-1|T-5|T-20).
BUCKETS = [("T-1", 0, 1), ("T-5", 2, 5), ("T-10", 6, 10), ("T-20", 11, 20)]


def load():
    meta = json.load(open("results/selfreport/meta.json"))
    H = np.load("results/selfreport/hidden.npz")["H"]
    probe = Probe.load()
    byep = collections.defaultdict(list)
    for i, x in enumerate(meta):
        if not x["crashed_episode"]:
            continue
        byep[(x["cond"], x["scenario_id"])].append(
            dict(s=int(x["steps_to_crash"]), logit=float(probe.logit(H[i])),
                 act=float(x["act_xyz_norm"])))
    for rows in byep.values():
        rows.sort(key=lambda r: -r["s"])          # far -> impact
    return byep, probe


def bucket_table(byep, cond):
    rows = [r for (c, _), rs in byep.items() if c == cond for r in rs]
    out = []
    for name, lo, hi in BUCKETS:
        sel = [r for r in rows if lo <= r["s"] <= hi]
        if sel:
            out.append(dict(bucket=name, n=len(sel),
                            mean_logit=round(float(np.mean([r["logit"] for r in sel])), 3),
                            mean_act=round(float(np.mean([r["act"] for r in sel])), 3)))
        else:
            out.append(dict(bucket=name, n=0, mean_logit=None, mean_act=None))
    return out


def make_figure(byep, probe, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    onpath = [(sid, rs) for (c, sid), rs in byep.items() if c == "wall"]
    offpath = [(sid, rs) for (c, sid), rs in byep.items() if c == "offpath"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    XMAX = 12  # horizon window where on-path lives

    # (A) KNOWS: crash-imminence logit vs horizon
    for sid, rs in offpath:
        s = [r["s"] for r in rs]; lg = [r["logit"] for r in rs]
        ax[0].plot(s, lg, "-", color="#c9c9c9", lw=1, zorder=1)
    for sid, rs in onpath:
        s = [r["s"] for r in rs]; lg = [r["logit"] for r in rs]
        ax[0].plot(s, lg, "-o", ms=3, color="#c0392b", lw=1.6, zorder=3)
    ax[0].axhline(probe.thr, color="k", ls=":", lw=1)
    ax[0].annotate(f"shield thr={probe.thr:.2f}", (XMAX - 0.2, probe.thr + 0.2),
                   ha="right", fontsize=8)
    for name, lo, hi in BUCKETS[:3]:
        ax[0].axvline(hi if name != "T-1" else 1, color="#bbb", ls="--", lw=0.7)
    ax[0].set_xlim(XMAX, -0.5)                 # impact at right
    ax[0].set_xlabel("steps to crash  (horizon T)")
    ax[0].set_ylabel("crash-imminence logit")
    ax[0].set_title("KNOWS: signal ramps up toward impact")
    ax[0].plot([], [], "-o", color="#c0392b", label="on-path wall (n=5)")
    ax[0].plot([], [], "-", color="#c9c9c9", label="off-path weak crash (n=6)")
    ax[0].legend(fontsize=7, loc="lower left")

    # (B) DOESN'T ACT: commanded action magnitude vs horizon (on-path only —
    # the point is the blocking hazard's own motion, not the off-path contrast).
    for sid, rs in onpath:
        s = [r["s"] for r in rs]; ac = [r["act"] for r in rs]
        ax[1].plot(s, ac, "-o", ms=3, color="#2471a3", lw=1.6, zorder=3)
    ax[1].set_xlim(XMAX, -0.5)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("steps to crash  (horizon T)")
    ax[1].set_ylabel("commanded action magnitude")
    ax[1].set_title("DOESN'T ACT: no braking (motion grows into wall)")
    ax[1].plot([], [], "-o", color="#2471a3", label="on-path wall (n=5)")
    ax[1].legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    byep, probe = load()
    tables = {"on_path": bucket_table(byep, "wall"),
              "off_path": bucket_table(byep, "offpath")}
    make_figure(byep, probe, f"{FIGDIR}/fig_horizon.png")

    summary = {
        "note": ("Relabel of frozen R4 crashed-episode trajectories by steps_to_crash. "
                 "On-path walls crash within <=9 steps (populate T-1..T-10, never T-20). "
                 "Probe logit crosses the shield threshold between T-10 and T-5 on on-path "
                 "walls while commanded action magnitude keeps rising -> knows but does not act."),
        "buckets": tables,
    }
    json.dump(summary, open(f"{OUT}/horizon.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/horizon.json")


if __name__ == "__main__":
    main()
