#!/usr/bin/env python
"""Is the "I will crash" direction shared across hazards, or hazard-specific?

The frozen WALL probe does not fire on glass (probe_glass_analysis.py: transfer AUC 0.36). That
alone is ambiguous — it could mean (a) the wall-crash direction genuinely differs from the
glass-crash direction, or (b) a mere domain/scale mismatch (glass frames out of the wall probe's
PCA basis). This resolves it with two cheap CPU tests over the two frozen captures:

  JOINT   — fit ONE PCA-50 + logreg on wall+glass frames pooled, leave-one-scenario-out across
            BOTH hazards. High AUC => a single linear direction decodes both (danger code shared).
  DIRECTED — train on one hazard, test on the other (both ways), same PCA/logreg pipeline as R4.
            This is the fair transfer test (refits PCA on the train hazard, unlike reusing the
            baked wall probe), isolating direction overlap from the baked probe's scale.

Reads results/selfreport/{hidden.npz,meta.json} (wall) + results/selfreport_glass/{...} (glass).
CPU-only. Appends to results/selfreport_glass/probe_glass_summary.json.
"""

from __future__ import annotations

import json
import numpy as np

PCA_K = 50


def auc(scores, labels):
    labels = np.asarray(labels).astype(bool); scores = np.asarray(scores)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pca_fit(X, k):
    mu = X.mean(0); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k].T


def logreg_fit(X, y, l2=2.0, iters=400, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))]); w = np.zeros(Xs.shape[1])
    y = np.asarray(y, float); n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xs @ w)); g = Xs.T @ (p - y) / n; g[:-1] += l2 * w[:-1] / n; w -= lr * g
    return (mu, sd, w)


def score(m, X):
    mu, sd, w = m
    return np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w


def load(kind):
    """Return H, scenario-id, positive-label(<=5 to crash), benign-mask for one hazard."""
    base = "results/selfreport" if kind == "wall" else "results/selfreport_glass"
    H = np.load(f"{base}/hidden.npz")["H"].astype(np.float32)
    meta = json.load(open(f"{base}/meta.json"))
    cond = np.array([m["cond"] for m in meta])
    sid = np.array([f"{kind}:" + m["scenario_id"] for m in meta])
    stc = np.array([m["steps_to_crash"] for m in meta])
    crashed = np.array([m["crashed_episode"] for m in meta])
    onpath = (cond == "wall") if kind == "wall" else (cond == "glass")
    benign = (cond == "nowall") if kind == "wall" else (cond == "noglass")
    pos = onpath & crashed & (stc >= 0) & (stc <= 5)
    keep = onpath | benign                     # drop off-path confound from the pooled fit
    return H[keep], sid[keep], pos[keep], benign[keep]


def loso_auc(H, sid, y):
    oof = np.full(len(y), np.nan)
    for held in np.unique(sid):
        tr, te = sid != held, sid == held
        if y[tr].sum() == 0 or (~y[tr]).sum() == 0:
            continue
        mu, V = pca_fit(H[tr], PCA_K)
        oof[te] = score(logreg_fit((H[tr] - mu) @ V, y[tr]), (H[te] - mu) @ V)
    v = ~np.isnan(oof)
    return auc(oof[v], y[v])


def directed_auc(Htr, ytr, Hte, yte):
    mu, V = pca_fit(Htr, PCA_K)
    m = logreg_fit((Htr - mu) @ V, ytr)
    return auc(score(m, (Hte - mu) @ V), yte)


def main():
    Hw, sw, yw, bw = load("wall")
    Hg, sg, yg, bg = load("glass")
    print(f"wall  frames={len(yw)} pos={int(yw.sum())}   glass frames={len(yg)} pos={int(yg.sum())}")

    # JOINT: one probe, LOSO across both hazards' scenarios
    Hj = np.vstack([Hw, Hg]); sj = np.concatenate([sw, sg]); yj = np.concatenate([yw, yg])
    a_joint = loso_auc(Hj, sj, yj)

    # DIRECTED transfer (fair: PCA refit on the train hazard)
    a_w2g = directed_auc(Hw, yw, Hg, yg)
    a_g2w = directed_auc(Hg, yg, Hw, yw)

    # within-hazard reference under the identical pipeline
    a_w = loso_auc(Hw, sw, yw)
    a_g = loso_auc(Hg, sg, yg)

    out = {
        "within_wall_loso_auc": round(a_w, 3),
        "within_glass_loso_auc": round(a_g, 3),
        "joint_loso_auc_both_hazards": round(a_joint, 3),
        "directed_train_wall_test_glass_auc": round(a_w2g, 3),
        "directed_train_glass_test_wall_auc": round(a_g2w, 3),
    }
    print(json.dumps(out, indent=2))

    p = "results/selfreport_glass/probe_glass_summary.json"
    s = json.load(open(p)); s["joint_and_directed_transfer"] = out
    json.dump(s, open(p, "w"), indent=2)
    print(f"appended -> {p}")


if __name__ == "__main__":
    main()
