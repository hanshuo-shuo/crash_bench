#!/usr/bin/env python
"""Figures for Path 1-1a (probe-triggered safe-abort). CPU-only (numpy + matplotlib).

Reads results/intervention/{episodes,summary}.json and writes:
  setup/figures/fig_intervention.png       — 3 panels: crash rate, peak force, logit traces
The logit-trace panel is the mechanism: the guard fires (crosses thr) a few steps before the
baseline would crash on on-path walls, while no-wall / off-path traces stay below thr (no fire).
"""

from __future__ import annotations

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = "results/intervention"
FIG = "setup/figures"
os.makedirs(FIG, exist_ok=True)

rows = json.load(open(f"{IN}/episodes.json"))
summary = json.load(open(f"{IN}/summary.json"))
thr = summary["headline"]["thr"]


def by(cond):
    return [r for r in rows if r["cond"] == cond]


fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# panel A: crash rate, baseline vs guarded (treatment)
conds = ["treatment_baseline", "treatment_guarded"]
cr = [summary[c]["crash_rate"] * 100 for c in conds]
bars = ax[0].bar(["baseline\nOpenVLA", "probe-guarded"], cr, color=["tab:red", "tab:green"])
for b, v, c in zip(bars, cr, conds):
    ax[0].text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%\n({summary[c]['n_crash']}/{summary[c]['n']})",
               ha="center", va="bottom", fontsize=10)
ax[0].set_ylim(0, 112); ax[0].set_ylabel("crash rate (%)")
ax[0].set_title("(1) Probe-guard eliminates the crash")

# panel B: peak contact force per episode (strip), baseline vs guarded
xb = np.full(len(by("treatment_baseline")), 0.0)
xg = np.full(len(by("treatment_guarded")), 1.0)
fb = [r["peak_contact_force"] for r in by("treatment_baseline")]
fg = [r["peak_contact_force"] for r in by("treatment_guarded")]
ax[1].scatter(xb + np.random.uniform(-.08, .08, len(xb)), fb, c="tab:red", s=40, alpha=.8, label="baseline")
ax[1].scatter(xg + np.random.uniform(-.08, .08, len(xg)), fg, c="tab:green", s=40, alpha=.8, label="guarded")
ax[1].axhline(75, color="gray", ls="--", lw=1, label="75 N crash threshold")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["baseline\nOpenVLA", "probe-guarded"])
ax[1].set_ylabel("peak contact force (N)")
ax[1].set_title("(2) Impact ~250 N -> ~0 N"); ax[1].legend(loc="upper right", fontsize=8)

# panel C: probe-logit traces (mechanism + confound)
def plot_traces(cond, color, label, lw=1.2, alpha=.5):
    first = True
    for r in by(cond):
        lg = [x for x in (r.get("logits") or []) if x is not None]
        if not lg:
            continue
        ax[2].plot(range(len(lg)), lg, color=color, lw=lw, alpha=alpha,
                   label=label if first else None)
        first = False

plot_traces("treatment_guarded", "tab:red", "on-path wall (fires)")
plot_traces("offpath_guarded", "tab:orange", "off-path wall")
plot_traces("nowall_guarded", "tab:green", "no wall")
ax[2].axhline(thr, color="black", ls="--", lw=1.4, label=f"trigger thr={thr:.1f}")
ax[2].set_xlabel("policy step"); ax[2].set_ylabel("probe crash-logit")
ax[2].set_title("(3) Fires on on-path crashes only")
ax[2].legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIG}/fig_intervention.png", dpi=130)
print(f"wrote {FIG}/fig_intervention.png")
