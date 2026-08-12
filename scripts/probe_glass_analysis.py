#!/usr/bin/env python
"""Self-report probe on Category 2 (glass): within-glass fit + cross-hazard transfer.

Reads results/selfreport_glass/{hidden.npz, meta.json} (from probe_glass_capture.py) and answers
two questions, mirroring R4 on the wall:

  (1) WITHIN-GLASS — is "I'm about to hit the glass" linearly decodable from OpenVLA's own hidden
      state?  PCA-50 + logistic regression, labeled "crash within T steps", leave-one-SCENARIO-out
      (no temporal leakage). Off-path control frames held out as the confound test.

  (2) CROSS-HAZARD TRANSFER — does the FROZEN WALL probe (results/selfreport/probe_T5.npz, never
      shown a glass) already fire on glass pre-collision frames?  If yes, the "I will crash"
      representation is hazard-agnostic, not wall-specific — the strongest form of the R4 claim.

CPU-only (numpy + matplotlib). Outputs setup/figures/fig_glass_probe.png + results/selfreport_glass/
probe_glass_summary.json.
"""

from __future__ import annotations

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crashbench.probe import Probe

IN = "results/selfreport_glass"
WALL_PROBE = "results/selfreport/probe_T5.npz"
FIG = "setup/figures"
os.makedirs(FIG, exist_ok=True)
T_LIST = [1, 3, 5, 10]
PCA_K = 50


# ---------- tiny numpy ML (identical to probe_selfreport_analysis.py) ----------
def auc(scores, labels):
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pca_fit(X, k):
    mu = X.mean(0); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:k].T


def logreg_fit(X, y, l2=1.0, iters=400, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xs.shape[1]); y = np.asarray(y, dtype=np.float64); n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xs @ w))
        g = Xs.T @ (p - y) / n
        g[:-1] += l2 * w[:-1] / n
        w -= lr * g
    return (mu, sd, w)


def logreg_score(model, X):
    mu, sd, w = model
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return Xs @ w


# ---------- load ----------
with np.load(f"{IN}/hidden.npz", allow_pickle=False) as archive:
    hidden_key = "hidden" if "hidden" in archive.files else "H"
    H = archive[hidden_key].astype(np.float32)
meta = json.load(open(f"{IN}/meta.json"))
cond = np.array([m["cond"] for m in meta])
sid = np.array([m["scenario_id"] for m in meta])
stc = np.array([m["steps_to_crash"] for m in meta])
crashed = np.array([m["crashed_episode"] for m in meta])
actn = np.array([m["act_xyz_norm"] for m in meta])
print(f"loaded H={H.shape}  conds={dict(zip(*np.unique(cond, return_counts=True)))}")

is_glass = cond == "glass"          # cup on the path (positive-bearing)
is_noglass = cond == "noglass"      # nominal reach (negative)
is_off = cond == "offpath"          # cup off the path (confound, held out)


def labels_for_T(T):
    # positive = an on-path glass frame within T steps of the actual strike
    return is_glass & crashed & (stc >= 0) & (stc <= T)


summary = {"n_frames": int(len(meta)),
           "frames_by_cond": {k: int(v) for k, v in zip(*np.unique(cond, return_counts=True))}}

# ---------- (1) WITHIN-GLASS linear probe, leave-one-scenario-out ----------
auc_by_T = {}
glass_oof_T5 = None
for T in T_LIST:
    y = labels_for_T(T)
    pool = is_glass | is_noglass                # offpath held out for the confound test
    scen_list = np.unique(sid[is_glass])
    oof = np.full(len(meta), np.nan)
    for held in scen_list:
        tr = pool & (sid != held); te = pool & (sid == held)
        if y[tr].sum() == 0 or (~y[tr]).sum() == 0:
            continue
        mu, V = pca_fit(H[tr], PCA_K)
        model = logreg_fit((H[tr] - mu) @ V, y[tr], l2=2.0)
        oof[te] = logreg_score(model, (H[te] - mu) @ V)
    valid = ~np.isnan(oof) & pool
    a = auc(oof[valid], y[valid])
    auc_by_T[T] = a
    print(f"  [within-glass] probe AUC (crash within {T:2d} steps, LOSO) = {a:.3f}")
    if T == 5:
        glass_oof_T5 = oof
summary["within_glass_probe_auc_by_T"] = {k: (None if np.isnan(v) else round(v, 3))
                                          for k, v in auc_by_T.items()}

# ---------- (2) CROSS-HAZARD TRANSFER: frozen WALL probe scores glass frames ----------
transfer = {"available": os.path.exists(WALL_PROBE)}
if transfer["available"]:
    wp = Probe.load(WALL_PROBE)
    wall_logit = np.array([wp.logit(h) for h in H], dtype=np.float32)
    yb = labels_for_T(5)                                   # glass pre-crash (<=5) = positive
    benign = is_noglass | is_off                           # nominal + off-path = negative
    tr_auc = auc(np.concatenate([wall_logit[yb], wall_logit[benign]]),
                 np.concatenate([np.ones(yb.sum()), np.zeros(benign.sum())]))
    fire_pre = float((wall_logit[yb] > wp.thr).mean()) if yb.sum() else float("nan")
    fire_noglass = float((wall_logit[is_noglass] > wp.thr).mean()) if is_noglass.sum() else float("nan")
    fire_off = float((wall_logit[is_off] > wp.thr).mean()) if is_off.sum() else float("nan")
    print(f"  [transfer] WALL-probe AUC on glass (pre-crash<=5 vs benign) = {tr_auc:.3f}")
    print(f"  [transfer] fire-rate @wall thr={wp.thr:.2f}: glass pre-crash={fire_pre:.1%}  "
          f"no-glass={fire_noglass:.1%}  off-path={fire_off:.1%}")
    transfer.update({"wall_probe_thr": round(float(wp.thr), 3),
                     "wall_probe_auc_on_glass_pre5": round(tr_auc, 3),
                     "fire_rate_glass_pre5": round(fire_pre, 3),
                     "fire_rate_noglass": round(fire_noglass, 3),
                     "fire_rate_offpath": round(fire_off, 3)})
else:
    print(f"  [transfer] SKIPPED — {WALL_PROBE} not found (run probe_selfreport_analysis.py first)")
summary["cross_hazard_transfer"] = transfer

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

# (left) within-glass out-of-fold logit vs steps-to-crash; braking check via action magnitude
if glass_oof_T5 is not None:
    gm = is_glass & crashed & (stc >= 0) & ~np.isnan(glass_oof_T5)
    ax[0].scatter(stc[gm], glass_oof_T5[gm], s=22, c="tab:red", alpha=.7, label="on-path glass (LOSO logit)")
    nm = is_noglass & ~np.isnan(glass_oof_T5)
    ax[0].scatter(np.zeros(nm.sum()) - 1, glass_oof_T5[nm], s=14, c="tab:green", alpha=.4,
                  label="no-glass (nominal)")
    ax[0].axvline(5, ls=":", c="gray", lw=1); ax[0].set_xlim(-2, 20)
ax[0].invert_xaxis()
ax[0].set_xlabel("steps until strike  (0 = impact)")
ax[0].set_ylabel("within-glass probe logit (out-of-fold)")
ax[0].set_title(f"(1) glass self-report — LOSO AUC@T5 = {auc_by_T.get(5, float('nan')):.3f}")
ax[0].legend(fontsize=8)

# (right) cross-hazard transfer: frozen wall-probe logit, glass pre-crash vs benign
if transfer["available"]:
    yb = labels_for_T(5)
    ax[1].hist(wall_logit[is_noglass], bins=25, alpha=.5, color="tab:green", label="no-glass (benign)")
    ax[1].hist(wall_logit[is_off], bins=25, alpha=.5, color="tab:blue", label="off-path (confound)")
    ax[1].hist(wall_logit[yb], bins=25, alpha=.6, color="tab:red", label="glass pre-crash (<=5)")
    ax[1].axvline(wp.thr, ls="--", c="k", lw=1.3, label=f"wall thr={wp.thr:.2f}")
    ax[1].set_xlabel("FROZEN wall-probe logit (never trained on glass)")
    ax[1].set_ylabel("frames")
    ax[1].set_title(f"(2) cross-hazard transfer — AUC = {transfer['wall_probe_auc_on_glass_pre5']:.3f}")
    ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_glass_probe.png", dpi=130)
print(f"wrote {FIG}/fig_glass_probe.png")

json.dump(summary, open(f"{IN}/probe_glass_summary.json", "w"), indent=2)
print(f"wrote {IN}/probe_glass_summary.json")
print(json.dumps(summary, indent=2))
