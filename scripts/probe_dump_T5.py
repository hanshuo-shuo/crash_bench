#!/usr/bin/env python
"""Dump ONLY the T=5 probe params to results/selfreport/probe_T5.npz (fast, CPU/numpy).

Same fit as scripts/probe_selfreport_analysis.py's full T=5 model (PCA-50 + L2 logreg on
the wall+nowall pool), but skips the slow 4-horizon LOSO loop and the figures. Used to
provision the probe for the Path 1-1a intervention without re-running the whole analysis.
"""
from __future__ import annotations
import json
import numpy as np

IN = "results/selfreport"
PCA_K = 50
T = 5


def pca_fit(X, k):
    mu = X.mean(0); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:k].T


def logreg_fit(X, y, l2=2.0, iters=400, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xs.shape[1]); y = np.asarray(y, np.float64); n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xs @ w))
        g = Xs.T @ (p - y) / n
        g[:-1] += l2 * w[:-1] / n
        w -= lr * g
    return mu, sd, w


def logreg_score(model, X):
    mu, sd, w = model
    return np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w


print("loading hidden states...", flush=True)
H = np.load(f"{IN}/hidden.npz")["H"].astype(np.float32)
meta = json.load(open(f"{IN}/meta.json"))
cond = np.array([m["cond"] for m in meta])
stc = np.array([m["steps_to_crash"] for m in meta])
crashed = np.array([m["crashed_episode"] for m in meta])
is_wall, is_nowall, is_off = cond == "wall", cond == "nowall", cond == "offpath"
y = is_wall & crashed & (stc >= 0) & (stc <= T)       # crash within T steps
pool = is_wall | is_nowall

print(f"H={H.shape}  pool={pool.sum()}  positives={y[pool].sum()}", flush=True)
print("PCA fit (one SVD)...", flush=True)
mu, V = pca_fit(H[pool], PCA_K)
print("logreg fit...", flush=True)
model = logreg_fit((H[pool] - mu) @ V, y[pool])
mu_lr, sd_lr, w_lr = model

# threshold on the NEGATIVE pool (off-path + no-wall) so the guard ~never false-fires.
off_logit = logreg_score(model, (H[is_off] - mu) @ V) if is_off.sum() else np.array([])
nowall_logit = logreg_score(model, (H[is_nowall] - mu) @ V)
neg = np.concatenate([off_logit, nowall_logit]) if len(off_logit) else nowall_logit
wall_logit = logreg_score(model, (H[is_wall] - mu) @ V)
onpath_med = float(np.median(wall_logit[y[is_wall]]))
thr = float(neg.max() + 0.10 * (onpath_med - neg.max()))

np.savez(f"{IN}/probe_T5.npz",
         mu_pca=mu.astype(np.float32), V_pca=V.astype(np.float32),
         mu_lr=mu_lr.astype(np.float32), sd_lr=sd_lr.astype(np.float32),
         w_lr=w_lr.astype(np.float32), thr=np.float32(thr), pca_k=np.int64(PCA_K))
print(f"wrote {IN}/probe_T5.npz", flush=True)
print(f"  thr={thr:.3f}  neg_max={neg.max():.2f}  neg_p99={np.percentile(neg,99):.2f}  "
      f"onpath<=5_median={onpath_med:.2f}", flush=True)
