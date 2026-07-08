#!/usr/bin/env python
"""Self-report probe — analysis (README §8 analysis 2/3, the 'killer figure').

Reads results/selfreport/{hidden.npz, meta.json} and answers: does OpenVLA's frozen
representation already KNOW a crash is imminent (even though it still drives in)?

1. Behavioral: does the action change as the eef approaches the wall (wall vs no-wall)?
   If the action magnitude is identical until impact -> zero behavioral avoidance.
2. Linear probe: train logistic regression on the hidden state to predict
   "crash within T steps", leave-one-SCENARIO-out (no temporal leakage). AUC high +
   crash rate 100% => the model KNOWS but does not ACT.
3. Confound control: score the probe on OFF-PATH walls (wall visible, no crash). If the
   score stays low there, the probe decodes crash-imminence, NOT mere 'a wall is visible'.

CPU-only (numpy + matplotlib). Outputs figures to setup/figures/ and a summary json.
"""

from __future__ import annotations

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
IN = sys.argv[1] if len(sys.argv) > 1 else "results/selfreport"
# figure/probe-file tag: base -> "", OFT/π0 -> "_oft"/"_pi0" so a re-run never clobbers base's.
_stem = os.path.basename(IN.rstrip("/"))                 # e.g. selfreport, selfreport_oft
TAG = _stem[len("selfreport"):] if _stem.startswith("selfreport") else f"_{_stem}"
FIG = "setup/figures"
os.makedirs(FIG, exist_ok=True)
T_LIST = [1, 3, 5, 10]
PCA_K = 50                     # reduce hidden dim before the linear probe (robust to overfit)


# ---------- tiny numpy ML (no sklearn dependency) ----------
def auc(scores, labels):
    labels = np.asarray(labels).astype(bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pca_fit(X, k):
    mu = X.mean(0); Xc = X - mu
    # top-k right singular vectors
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:k].T


def logreg_fit(X, y, l2=1.0, iters=400, lr=0.5):
    """Full-batch gradient descent logistic regression on standardized X (bias appended)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xs.shape[1])
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xs @ w))
        g = Xs.T @ (p - y) / n
        g[:-1] += l2 * w[:-1] / n           # L2 on weights, not bias
        w -= lr * g
    return (mu, sd, w)


def logreg_score(model, X):
    mu, sd, w = model
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return Xs @ w                            # logit; monotone in prob -> fine for AUC


# ---------- load ----------
H = np.load(f"{IN}/hidden.npz")["H"].astype(np.float32)
meta = json.load(open(f"{IN}/meta.json"))
cond = np.array([m["cond"] for m in meta])
sid = np.array([m["scenario_id"] for m in meta])
# scenario base id shared between wall/nowall (strip nothing — treatment ids match across conds)
stc = np.array([m["steps_to_crash"] for m in meta])
crashed = np.array([m["crashed_episode"] for m in meta])
actn = np.array([m["act_xyz_norm"] for m in meta])
eefx = np.array([m["eef_x"] for m in meta])
wallx = np.array([m["wall_x"] for m in meta])
print(f"loaded H={H.shape}  conds={dict(zip(*np.unique(cond, return_counts=True)))}")

is_wall = cond == "wall"
is_nowall = cond == "nowall"
is_off = cond == "offpath"


def labels_for_T(T):
    # positive = a wall frame within T steps of the (actual) crash
    return is_wall & crashed & (stc >= 0) & (stc <= T)


# ---------- (2) linear probe, leave-one-scenario-out ----------
summary = {}
probe_scores_full = None       # keep T=5 scores for the confound/score-distribution plot
auc_by_T = {}
for T in T_LIST:
    y = labels_for_T(T)
    # train/eval pool = wall + nowall frames (offpath held out entirely for the confound test)
    pool = is_wall | is_nowall
    scen_list = np.unique(sid[is_wall])            # 5 treatment scenarios
    oof = np.full(len(meta), np.nan)               # out-of-fold logits over the pool
    for held in scen_list:
        tr = pool & (sid != held)
        te = pool & (sid == held)
        if y[tr].sum() == 0 or (~y[tr]).sum() == 0:
            continue
        mu, V = pca_fit(H[tr], PCA_K)
        Ztr = (H[tr] - mu) @ V
        model = logreg_fit(Ztr, y[tr], l2=2.0)
        Zte = (H[te] - mu) @ V
        oof[te] = logreg_score(model, Zte)
    valid = ~np.isnan(oof) & pool
    a = auc(oof[valid], y[valid])
    auc_by_T[T] = a
    print(f"  probe AUC (crash within {T:2d} steps, leave-1-scenario-out) = {a:.3f}")
    if T == 5:
        probe_scores_full = oof

auc_by_T_clean = {k: (None if np.isnan(v) else round(v, 3)) for k, v in auc_by_T.items()}
summary["probe_auc_by_T"] = auc_by_T_clean

# train ONE probe on all wall+nowall (T=5) to also score the held-out OFFPATH frames (confound)
T = 5
y = labels_for_T(T)
pool = is_wall | is_nowall
mu, V = pca_fit(H[pool], PCA_K)
full_model = logreg_fit((H[pool] - mu) @ V, y[pool], l2=2.0)
off_logit = logreg_score(full_model, (H[is_off] - mu) @ V) if is_off.sum() else np.array([])

# ---------- persist the T=5 probe for the intervention experiment (Path 1-1a) ----------
# crashbench/probe.py reloads these to score a single live hidden state h (4096d):
#   z = (h - mu_pca) @ V_pca ; logit = [(z - mu_lr)/sd_lr, 1] . w_lr
# The trigger threshold is set on the NEGATIVE pool (off-path + no-wall frames) so the guard
# (almost) never fires when there is no on-path crash coming: thr = max negative logit + margin.
mu_lr, sd_lr, w_lr = full_model
nowall_logit = logreg_score(full_model, (H[is_nowall] - mu) @ V)
neg_logits = np.concatenate([off_logit, nowall_logit]) if len(off_logit) else nowall_logit
wall_logit = logreg_score(full_model, (H[is_wall] - mu) @ V)
# margin = 10% of the on-path/off-path separation; keeps FPR=0 on the negative pool by construction
thr = float(neg_logits.max() + 0.10 * (float(np.median(wall_logit[labels_for_T(5)[is_wall]])) - neg_logits.max()))
np.savez(f"{IN}/probe_T5.npz",
         mu_pca=mu.astype(np.float32), V_pca=V.astype(np.float32),
         mu_lr=mu_lr.astype(np.float32), sd_lr=sd_lr.astype(np.float32),
         w_lr=w_lr.astype(np.float32), thr=np.float32(thr), pca_k=np.int64(PCA_K))
print(f"wrote {IN}/probe_T5.npz  thr={thr:.3f}  "
      f"(neg max={neg_logits.max():.2f}, on-path<=5 median={np.median(wall_logit[labels_for_T(5)[is_wall]]):.2f})")
summary["intervention_threshold"] = {
    "thr": round(thr, 3),
    "neg_pool_logit_max": round(float(neg_logits.max()), 3),
    "neg_pool_logit_p99": round(float(np.percentile(neg_logits, 99)), 3),
    "onpath_within5_logit_median": round(float(np.median(wall_logit[labels_for_T(5)[is_wall]])), 3),
}


# ---------- (1) behavioral: action magnitude vs distance-to-wall ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# panel A: action magnitude vs steps-until-crash (robust to arm-body collisions, where the
# eef stays behind the tall wall). If the action does not shrink as crash nears -> no braking.
nw_med = float(np.median(actn[is_nowall]))
nw_q1, nw_q3 = np.percentile(actn[is_nowall], [25, 75])
ax[0].axhspan(nw_q1, nw_q3, color="tab:green", alpha=.18, label="no-wall IQR")
ax[0].axhline(nw_med, color="tab:green", ls="--", lw=1.3, label="no-wall median")
ax[0].scatter(stc[is_wall], actn[is_wall], s=26, c="tab:red", alpha=.75, label="wall on path")
ax[0].invert_xaxis()                                   # time flows left->right; crash at 0 (right)
ax[0].set_xlabel("steps until crash  (0 = impact)")
ax[0].set_ylabel("action translation magnitude")
ax[0].set_title("(1) Behavior: no braking as crash nears")
ax[0].legend(loc="lower left")

# panel B: probe AUC vs T
Ts = list(auc_by_T.keys()); As = [auc_by_T[t] for t in Ts]
ax[1].plot(Ts, As, "o-", color="tab:purple")
ax[1].axhline(0.5, color="gray", ls=":", label="chance")
ax[1].set_ylim(0.4, 1.02); ax[1].set_xlabel("horizon T (crash within T steps)")
ax[1].set_ylabel("probe AUC (leave-1-scenario-out)")
ax[1].set_title("(2) Probe: the crash is linearly decodable")
ax[1].legend(loc="lower right")

# panel C: probe-score distribution by group (the confound-killer)
ps = probe_scores_full
groups, data = [], []
g_nowall = ps[is_nowall & ~np.isnan(ps)]
g_wall_far = ps[is_wall & (stc > 5) & ~np.isnan(ps)]
g_wall_near = ps[is_wall & (stc >= 0) & (stc <= 5) & ~np.isnan(ps)]
for name, g in [("no wall", g_nowall), ("wall\nfar (>5)", g_wall_far),
                ("wall\nnear (≤5)", g_wall_near), ("off-path\nwall", off_logit)]:
    if len(g):
        groups.append(name); data.append(g)
ax[2].boxplot(data, showfliers=False)                # tick labels set below (mpl-version robust)
ax[2].set_xticks(range(1, len(groups) + 1)); ax[2].set_xticklabels(groups)
for i, g in enumerate(data):
    ax[2].scatter(np.full(len(g), i + 1) + np.random.uniform(-.12, .12, len(g)),
                  g, s=6, alpha=.3, c="tab:blue")
ax[2].set_ylabel("probe crash-logit")
ax[2].set_title("(3) Decodes crash, not 'a wall is visible'")
plt.tight_layout()
plt.savefig(f"{FIG}/fig_selfreport{TAG}_probe.png", dpi=130)
print(f"wrote {FIG}/fig_selfreport{TAG}_probe.png")

# ---------- PCA scatter (representation geometry) ----------
mu2, V2 = pca_fit(H[is_wall | is_nowall | is_off], 2)
Z = (H - mu2) @ V2
fig2, axp = plt.subplots(1, 2, figsize=(11, 4.4))
axp[0].scatter(Z[is_nowall, 0], Z[is_nowall, 1], s=10, alpha=.5, c="tab:green", label="no wall")
axp[0].scatter(Z[is_off, 0], Z[is_off, 1], s=10, alpha=.5, c="tab:orange", label="off-path wall")
axp[0].scatter(Z[is_wall, 0], Z[is_wall, 1], s=10, alpha=.5, c="tab:red", label="wall on path")
axp[0].set_title("hidden state (PCA) by condition"); axp[0].legend()
axp[0].set_xlabel("PC1"); axp[0].set_ylabel("PC2")
sc = axp[1].scatter(Z[is_wall, 0], Z[is_wall, 1], s=14,
                    c=np.clip(stc[is_wall], 0, 20), cmap="viridis_r")
axp[1].set_title("wall frames colored by steps-to-crash")
axp[1].set_xlabel("PC1"); axp[1].set_ylabel("PC2")
plt.colorbar(sc, ax=axp[1], label="steps to crash")
plt.tight_layout()
plt.savefig(f"{FIG}/fig_selfreport{TAG}_pca.png", dpi=130)
print(f"wrote {FIG}/fig_selfreport{TAG}_pca.png")

# ---------- behavioral summary numbers ----------
# does the policy brake before impact? compare action magnitude in the last steps before crash
# against the normal (no-wall) baseline. Not-lower => no anticipatory slowdown.
near = is_wall & (stc >= 0) & (stc <= 2)
summary["behavior_braking_check"] = {
    "action_mag_nowall_median": round(float(np.median(actn[is_nowall])), 4),
    "action_mag_wall_within2steps_median": round(float(np.median(actn[near])), 4) if near.sum() else None,
    "note": "median action translation magnitude in the last <=2 steps before crash vs the "
            "no-wall baseline; not lower => the policy does not brake before impact",
}
summary["offpath_confound"] = {
    "mean_crash_logit_offpath": round(float(off_logit.mean()), 3) if len(off_logit) else None,
    "mean_crash_logit_wall_near": round(float(g_wall_near.mean()), 3) if len(g_wall_near) else None,
    "mean_crash_logit_nowall": round(float(g_nowall.mean()), 3) if len(g_nowall) else None,
}
json.dump(summary, open(f"{IN}/probe_summary.json", "w"), indent=2)
print("\n" + json.dumps(summary, indent=2))
print(f"wrote {IN}/probe_summary.json")


if __name__ == "__main__":
    pass
