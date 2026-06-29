#!/usr/bin/env python
"""Figure for Path 1-1b (activation steering — NULL result + mechanism). CPU-only.
Reads results/steering/{summary,diag}.json, writes setup/figures/fig_steering.png:
  (1) alpha sweep: treatment crash rate & peak force are flat -> steering does not brake
  (2) readout projection: the probe direction barely touches the action-token logits
      (||W_action @ d|| << ||W_full @ d||) -> WHY it is inert; plus ||da|| vs alpha
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = "results/steering"
FIG = "setup/figures"
os.makedirs(FIG, exist_ok=True)
s = json.load(open(f"{IN}/summary.json"))
diag = json.load(open(f"{IN}/diag.json"))
A = s["alphas"]
tr, nw = s["treatment"], s["nowall"]
def g(d, a, f):
    return d[str(a)][f] if str(a) in d else d[a][f]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# (1) sweep: treatment peak force + crash rate vs alpha (both flat -> null)
pf = [g(tr, a, "peak_force_mean") for a in A]
cr = [g(tr, a, "crash_rate") * 100 for a in A]
ax[0].plot(A, pf, "o-", color="tab:red", label="peak force (N)")
ax[0].axhline(75, color="gray", ls="--", lw=1, label="75 N crash thr")
ax[0].set_xlabel("steering alpha"); ax[0].set_ylabel("treatment peak force (N)", color="tab:red")
ax[0].tick_params(axis="y", labelcolor="tab:red"); ax[0].set_ylim(0, max(pf) * 1.2)
ax0b = ax[0].twinx()
ax0b.plot(A, cr, "s--", color="tab:purple", alpha=.7, label="crash rate (%)")
ax0b.set_ylabel("crash rate (%)", color="tab:purple"); ax0b.set_ylim(-5, 105)
ax0b.tick_params(axis="y", labelcolor="tab:purple")
ax[0].set_title("(1) NULL: steering doesn't brake\n(crash 100%, force flat at all alpha)")
ax[0].legend(loc="lower left", fontsize=8)

# (2) action displacement vs alpha (hook fires, but action moves only at absurd alpha)
da = [r["da_vs_alpha0"] for r in diag["action_shift_vs_alpha"]]
das = [r["alpha"] for r in diag["action_shift_vs_alpha"]]
ax[1].plot(das, da, "o-", color="tab:blue")
ax[1].set_xscale("symlog")
ax[1].set_xlabel("steering alpha (symlog)")
ax[1].set_ylabel("||action(alpha) - action(0)||")
ax[1].set_title("(2) Hook fires (not a bug):\naction moves only at absurd alpha, not toward braking")

# (3) readout projection: WHY it's inert — d barely touches the action-token logits
pf_full = diag["readout_projection"]["W_full_dot_d_norm"]
pf_act = diag["readout_projection"]["W_action256_dot_d_norm"]
bars = ax[2].bar(["action-token\nlogits (256)", "full vocab\nlogits"], [pf_act, pf_full],
                 color=["tab:red", "tab:gray"])
for b, v in zip(bars, [pf_act, pf_full]):
    ax[2].text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", fontsize=11)
ax[2].set_ylabel("||W @ d_unit||  (logit shift / unit alpha)")
ax[2].set_title("(3) WHY: crash direction is ~orthogonal\nto the action readout (0.74 of 7.69)")

plt.tight_layout()
plt.savefig(f"{FIG}/fig_steering.png", dpi=130)
print(f"wrote {FIG}/fig_steering.png")
