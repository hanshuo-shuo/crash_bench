#!/usr/bin/env python
"""Diagnose task-phase, pose, action, and wall-visibility confounds.

This is a read-only analysis of an existing self-report capture.  It never runs a
policy and never overwrites the capture or the frozen probe.  Inputs are the
row-aligned ``hidden.npz``/``meta.json`` pair produced by
``probe_selfreport_capture.py`` plus the already-fitted ``probe_T5.npz``.

The analysis has two matching audits:

1. Every on-path wall frame with 0 <= steps_to_crash <= T is paired with the
   nearest-timestep safe off-path frame and nearest-timestep no-wall frame.
2. The same on-path frames are paired with nearest neighbours in standardized
   (timestep, EEF xyz, action-magnitude) space.

It also fits identical L2 logistic probes under leave-one-on-path-scenario-out
cross-validation for timestep, EEF xyz, timestep+EEF xyz,
timestep+EEF xyz+action magnitude, and hidden state.  Hidden state uses the
same PCA-50 -> standardized logistic regression pipeline as the existing probe;
the low-dimensional covariates use the same logistic regression directly.

Outputs are written only under ``--output`` (by default
``results/task_phase_confound``):

  task_phase_confound.json
  task_phase_confound_summary.png

The output directory is intentionally separate from results/selfreport so the
original capture, probe, summary, and figures remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = "results/selfreport"
DEFAULT_PROBE = "results/selfreport/probe_T5.npz"
DEFAULT_OUTPUT = "results/task_phase_confound"
DEFAULT_T = 5
PCA_K = 50
PCA_OVERSAMPLE = 10
PCA_POWER_ITERATIONS = 2
PCA_SEED = 0
L2 = 2.0
LOGREG_ITERS = 400
LOGREG_LR = 0.5


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC, with stable average handling for ties."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float("nan")
    # Average ranks for ties.  This is small enough to keep a simple numpy-only
    # implementation while matching the usual Mann-Whitney definition.
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision with score ties grouped at the same threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order].astype(np.int64)
    cum_pos = 0
    cum_n = 0
    ap = 0.0
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and s[end] == s[start]:
            end += 1
        group_pos = int(y[start:end].sum())
        prev_recall = cum_pos / n_pos
        cum_pos += group_pos
        cum_n += end - start
        if group_pos:
            ap += (cum_pos / cum_n) * (cum_pos / n_pos - prev_recall)
        start = end
    return float(ap)


def pca_fit(x: np.ndarray, k: int = PCA_K) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic PCA with an exact small-problem path.

    The existing probe was already fitted and is used as-is for matching.  For
    the five new hidden-state LOSO fits, a full 4096 x 4096 SVD is needlessly
    expensive on the CPU node.  This fixed-seed randomized range finder only
    uses the training fold, then does an exact SVD on the small projected
    matrix.  It is deterministic and keeps the same PCA-50 + logistic model
    comparison across folds.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    kk = min(k, xc.shape[0], xc.shape[1])
    if xc.shape[1] <= 128 or kk >= min(xc.shape):
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        return mu, vt[:kk].T
    rank = min(kk + PCA_OVERSAMPLE, xc.shape[1], xc.shape[0])
    rng = np.random.default_rng(PCA_SEED)
    omega = rng.standard_normal((xc.shape[1], rank))
    q, _ = np.linalg.qr(xc @ omega, mode="reduced")
    for _ in range(PCA_POWER_ITERATIONS):
        q, _ = np.linalg.qr(xc.T @ q, mode="reduced")
        q, _ = np.linalg.qr(xc @ q, mode="reduced")
    _, _, vt = np.linalg.svd(q.T @ xc, full_matrices=False)
    return mu, vt[:kk].T


def logreg_fit(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = L2,
    iters: int = LOGREG_ITERS,
    lr: float = LOGREG_LR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the repository's numpy-only standardized L2 logistic regression."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    xs = np.hstack([(x - mu) / sd, np.ones((len(x), 1), dtype=np.float64)])
    w = np.zeros(xs.shape[1], dtype=np.float64)
    n = len(y)
    for _ in range(iters):
        z = np.clip(xs @ w, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-z))
        g = xs.T @ (p - y) / n
        g[:-1] += l2 * w[:-1] / n
        w -= lr * g
    return mu, sd, w


def logreg_score(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    mu, sd, w = model
    return np.hstack([(np.asarray(x, dtype=np.float64) - mu) / sd, np.ones((len(x), 1))]) @ w


def frozen_probe_score(
    h: np.ndarray,
    params: dict[str, np.ndarray],
) -> np.ndarray:
    """Score hidden rows with the existing, frozen T=5 probe."""
    mu = params["mu_pca"].astype(np.float32)
    v = params["V_pca"].astype(np.float32)
    mu_lr = params["mu_lr"].astype(np.float32)
    sd_lr = params["sd_lr"].astype(np.float32)
    w = params["w_lr"].astype(np.float32)
    z = (np.asarray(h, dtype=np.float32) - mu) @ v
    return np.hstack([(z - mu_lr) / sd_lr, np.ones((len(z), 1), dtype=np.float32)]) @ w


def finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def clean_json(value: Any) -> Any:
    """Convert numpy scalars/arrays and non-finite values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite_float(value)
    if isinstance(value, float):
        return finite_float(value)
    return value


def quantiles(x: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"q25": None, "median": None, "q75": None}
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    return {"q25": float(q25), "median": float(med), "q75": float(q75)}


def summarize_pair(
    on_logits: np.ndarray,
    off_logits: np.ndarray,
    nowall_logits: np.ndarray,
    off_dt: np.ndarray,
    nowall_dt: np.ndarray,
    off_dist: np.ndarray | None = None,
    nowall_dist: np.ndarray | None = None,
) -> dict[str, Any]:
    def one(control: np.ndarray, dt: np.ndarray, dist: np.ndarray | None) -> dict[str, Any]:
        delta = on_logits - control
        out: dict[str, Any] = {
            "n": int(len(control)),
            "onpath_collision_window_logit": quantiles(on_logits),
            "control_logit": quantiles(control),
            "paired_delta_onpath_minus_control": quantiles(delta),
            "mean_onpath_logit": float(np.mean(on_logits)),
            "mean_control_logit": float(np.mean(control)),
            "mean_paired_delta": float(np.mean(delta)),
            "mean_abs_timestep_difference": float(np.mean(np.abs(dt))),
            "max_abs_timestep_difference": int(np.max(np.abs(dt))) if len(dt) else None,
        }
        if dist is not None:
            out["nn_distance"] = quantiles(dist)
        return out

    return {
        "onpath": quantiles(on_logits),
        "offpath_safe": one(off_logits, off_dt, off_dist),
        "nowall": one(nowall_logits, nowall_dt, nowall_dist),
    }


def row_record(meta: list[dict[str, Any]], i: int, logit: float, distance: float | None = None) -> dict[str, Any]:
    m = meta[i]
    out = {
        "row_index": int(i),
        "cond": m["cond"],
        "scenario_id": m["scenario_id"],
        "t": int(m["t"]),
        "steps_to_crash": int(m["steps_to_crash"]),
        "eef_xyz": [float(m[k]) for k in ("eef_x", "eef_y", "eef_z")],
        "action_magnitude": float(m["act_xyz_norm"]),
        "probe_logit": float(logit),
    }
    if distance is not None:
        out["match_distance_standardized"] = float(distance)
    return out


def select_nearest_timestep(meta: list[dict[str, Any]], source_i: int, candidates: np.ndarray) -> int:
    t = int(meta[source_i]["t"])
    # np.lexsort gives deterministic tie-breaking: |dt|, scenario id, row index.
    order = sorted(
        (int(i) for i in candidates),
        key=lambda i: (abs(int(meta[i]["t"]) - t), str(meta[i]["scenario_id"]), i),
    )
    if not order:
        raise ValueError("empty candidate set for timestep matching")
    return order[0]


def feature_matrix(meta: list[dict[str, Any]], indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [meta[i]["t"], meta[i]["eef_x"], meta[i]["eef_y"], meta[i]["eef_z"], meta[i]["act_xyz_norm"]]
            for i in indices
        ],
        dtype=np.float64,
    )


def select_nearest_feature(
    source_feature: np.ndarray,
    candidate_features: np.ndarray,
    candidates: np.ndarray,
) -> tuple[int, float]:
    distances = np.linalg.norm(candidate_features - source_feature[None, :], axis=1)
    # candidates are passed in original row order; argmin is deterministic for ties.
    j = int(np.argmin(distances))
    return int(candidates[j]), float(distances[j])


def build_pair_records(
    meta: list[dict[str, Any]],
    h: np.ndarray,
    probe: dict[str, np.ndarray],
    source: np.ndarray,
    off_candidates: np.ndarray,
    nowall_candidates: np.ndarray,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return summary and per-frame pairs for one matching mode."""
    # Selected rows are small; scoring only these rows avoids a needless full
    # 5k x 4096 projection while preserving the exact frozen-probe calculation.
    selected = []
    pair_indices: list[tuple[int, int, int]] = []
    distances: list[tuple[float, float]] = []

    if mode == "timestep":
        for i in source:
            oi = select_nearest_timestep(meta, int(i), off_candidates)
            ni = select_nearest_timestep(meta, int(i), nowall_candidates)
            pair_indices.append((int(i), oi, ni))
            distances.append((float(abs(meta[oi]["t"] - meta[i]["t"])), float(abs(meta[ni]["t"] - meta[i]["t"]))))
    elif mode == "timestep_eef_action_nn":
        all_match = np.concatenate([source, off_candidates, nowall_candidates])
        scales = feature_matrix(meta, all_match).std(axis=0)
        scales[scales < 1e-8] = 1.0
        source_f = feature_matrix(meta, source)
        off_f = feature_matrix(meta, off_candidates)
        nowall_f = feature_matrix(meta, nowall_candidates)
        source_f = source_f / scales
        off_f = off_f / scales
        nowall_f = nowall_f / scales
        for j, i in enumerate(source):
            oi, od = select_nearest_feature(source_f[j], off_f, off_candidates)
            ni, nd = select_nearest_feature(source_f[j], nowall_f, nowall_candidates)
            pair_indices.append((int(i), oi, ni))
            distances.append((od, nd))
    else:
        raise ValueError(mode)

    for triplet in pair_indices:
        selected.extend(triplet)
    unique_selected = np.asarray(sorted(set(selected)), dtype=np.int64)
    selected_logits = frozen_probe_score(h[unique_selected], probe)
    score_by_index = {int(i): float(s) for i, s in zip(unique_selected, selected_logits)}

    on_logits = np.asarray([score_by_index[i] for i, _, _ in pair_indices])
    off_logits = np.asarray([score_by_index[o] for _, o, _ in pair_indices])
    nowall_logits = np.asarray([score_by_index[n] for _, _, n in pair_indices])
    off_dt = np.asarray([abs(meta[o]["t"] - meta[i]["t"]) for i, o, _ in pair_indices], dtype=np.float64)
    nowall_dt = np.asarray([abs(meta[n]["t"] - meta[i]["t"]) for i, _, n in pair_indices], dtype=np.float64)
    off_dist = np.asarray([d[0] for d in distances], dtype=np.float64) if mode != "timestep" else None
    nowall_dist = np.asarray([d[1] for d in distances], dtype=np.float64) if mode != "timestep" else None

    pairs = []
    for k, (i, o, n) in enumerate(pair_indices):
        pairs.append(
            {
                "onpath": row_record(meta, i, on_logits[k]),
                "offpath_safe": row_record(meta, o, off_logits[k], distances[k][0] if mode != "timestep" else None),
                "nowall": row_record(meta, n, nowall_logits[k], distances[k][1] if mode != "timestep" else None),
                "abs_timestep_difference": {
                    "offpath_safe": int(off_dt[k]),
                    "nowall": int(nowall_dt[k]),
                },
            }
        )
    details = {
        "matching": mode,
        "n_pairs": len(pairs),
        "pairs": pairs,
    }
    return summarize_pair(on_logits, off_logits, nowall_logits, off_dt, nowall_dt, off_dist, nowall_dist), details


def prepare_loso_data(
    meta: list[dict[str, Any]],
    h: np.ndarray,
    t: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    cond = np.asarray([m["cond"] for m in meta])
    crashed = np.asarray([bool(m["crashed_episode"]) for m in meta])
    stc = np.asarray([int(m["steps_to_crash"]) for m in meta])
    # Off-path is used as a visibility control only when that episode did not
    # itself crash.  The few boundary off-path crashes are counted and excluded
    # so the negative class stays a clean "visible wall, no collision" control.
    offpath_safe = (cond == "offpath") & ~crashed
    wall = cond == "wall"
    nowall = cond == "nowall"
    positive = wall & crashed & (stc >= 0) & (stc <= t)
    pool = wall | nowall | offpath_safe
    labels = positive[pool]
    indices = np.flatnonzero(pool)
    # The five wall scenario IDs are shared by the corresponding no-wall runs.
    # Off-path scenarios have their own IDs and remain available as visibility
    # controls in every fold.
    sid = np.asarray([str(m["scenario_id"]) for m in meta])
    groups = sorted(set(sid[wall]))
    counts = {
        "pool_rows": int(pool.sum()),
        "positive_rows": int(positive[pool].sum()),
        "negative_rows": int((~positive[pool]).sum()),
        "offpath_safe_rows": int(offpath_safe.sum()),
        "offpath_episode_rows_excluded_due_to_crash": int(((cond == "offpath") & crashed).sum()),
    }
    return indices, labels, sid, groups, counts


def fit_loso_feature(
    h: np.ndarray,
    meta: list[dict[str, Any]],
    indices: np.ndarray,
    labels: np.ndarray,
    sid: np.ndarray,
    groups: list[str],
    kind: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run identical LOSO logistic fitting and return metrics plus OOF logits."""
    # Work in original row indices so the held-out scenario exclusion is clear.
    oof = np.full(len(indices), np.nan, dtype=np.float64)
    x_all = feature_matrix(meta, indices)
    if kind == "timestep":
        x_all = x_all[:, [0]]
    elif kind == "eef_xyz":
        x_all = x_all[:, 1:4]
    elif kind == "timestep_eef_xyz":
        x_all = x_all[:, 0:4]
    elif kind == "timestep_eef_xyz_action_magnitude":
        x_all = x_all[:, 0:5]
    elif kind == "hidden_state":
        x_all = h[indices].astype(np.float32)
    else:
        raise ValueError(kind)

    fold_rows = []
    for held in groups:
        # Every wall/nowall row for the held scenario is withheld.  Off-path
        # rows do not share those scenario IDs and remain controls in each fold.
        train = sid[indices] != held
        test = sid[indices] == held
        if not test.any() or labels[train].sum() == 0 or (~labels[train]).sum() == 0:
            continue
        if kind == "hidden_state":
            mu_pca, v_pca = pca_fit(x_all[train], PCA_K)
            xtr = (x_all[train] - mu_pca) @ v_pca
            xte = (x_all[test] - mu_pca) @ v_pca
            model = logreg_fit(xtr, labels[train].astype(np.float64), l2=L2)
            oof[test] = logreg_score(model, xte)
            dim = int(v_pca.shape[1])
        else:
            model = logreg_fit(x_all[train], labels[train].astype(np.float64), l2=L2)
            oof[test] = logreg_score(model, x_all[test])
            dim = int(x_all.shape[1])
        fold_rows.append({"held_out_scenario": held, "n_train": int(train.sum()), "n_test": int(test.sum()), "feature_dim": dim})

    valid = np.isfinite(oof)
    result = {
        "roc_auc": auc(oof[valid], labels[valid]),
        "auprc": average_precision(oof[valid], labels[valid]),
        "n_evaluated": int(valid.sum()),
        "n_positive_evaluated": int(labels[valid].sum()),
        "positive_prevalence": float(labels[valid].mean()) if valid.any() else None,
        "folds": fold_rows,
        "classifier": {
            "model": "standardized L2 logistic regression, full-batch gradient descent",
            "l2": L2,
            "iterations": LOGREG_ITERS,
            "learning_rate": LOGREG_LR,
            "hidden_preprocessing": "training-fold deterministic randomized PCA-50 (seed 0, 2 power iterations) before the same logistic regression",
        },
    }
    return result, oof


def make_figure(
    output: Path,
    source_meta: list[dict[str, Any]],
    timestep_summary: dict[str, Any],
    nn_summary: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: paired frozen-probe logit by remaining steps.
    def plot_by_k(summary: dict[str, Any], title: str, axis: Any) -> None:
        pairs = summary["pairs"]
        ks = sorted(set(int(p["onpath"]["steps_to_crash"]) for p in pairs))
        labels = [("onpath collision", "tab:red"), ("offpath safe", "tab:orange"), ("no wall", "tab:green")]
        pair_key = {
            "onpath collision": "onpath",
            "offpath safe": "offpath_safe",
            "no wall": "nowall",
        }
        for name, color in labels:
            vals = []
            for k in ks:
                rows = [p[pair_key[name]] for p in pairs if int(p["onpath"]["steps_to_crash"]) == k]
                vals.append(float(np.mean([r["probe_logit"] for r in rows])))
            axis.plot(ks, vals, "o-", color=color, label=name)
        axis.set_xticks(ks)
        axis.set_xlabel("steps to crash (0 = impact)")
        axis.set_ylabel("frozen T=5 probe logit")
        axis.set_title(title)
        axis.grid(alpha=0.2)

    plot_by_k(timestep_summary["details"], "Timestep-matched probe logit", ax[0])
    ax[0].legend(fontsize=8)

    # Panel 2: matched group distributions, using the NN pairs.
    groups = ["onpath\ncollision", "offpath\nsafe", "no wall"]
    data = []
    for key in ["onpath", "offpath_safe", "nowall"]:
        data.append([p[key]["probe_logit"] for p in nn_summary["details"]["pairs"]])
    ax[1].boxplot(data, labels=groups, showfliers=False)
    for j, vals in enumerate(data, start=1):
        jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) else np.array([])
        ax[1].scatter(np.full(len(vals), j) + jitter, vals, s=13, alpha=0.55)
    ax[1].set_ylabel("frozen T=5 probe logit")
    ax[1].set_title("NN matched on timestep + EEF xyz + action")
    ax[1].grid(axis="y", alpha=0.2)

    # Panel 3: common LOSO classifier metrics.
    names = list(metrics)
    short = ["t", "EEF", "t+EEF", "t+EEF+a", "hidden"]
    x = np.arange(len(names))
    width = 0.36
    ax[2].bar(x - width / 2, [metrics[n]["roc_auc"] for n in names], width, label="ROC-AUC", color="tab:blue")
    ax[2].bar(x + width / 2, [metrics[n]["auprc"] for n in names], width, label="AUPRC", color="tab:purple")
    ax[2].axhline(0.5, color="gray", ls=":", lw=1)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(short, rotation=20)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("LOSO score")
    ax[2].set_title("Collision-window classification")
    ax[2].legend(fontsize=8)
    ax[2].grid(axis="y", alpha=0.2)

    fig.suptitle("Task-phase / pose / action confound diagnostic", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_INPUT, help="capture directory containing hidden.npz and meta.json")
    ap.add_argument("--probe", default=DEFAULT_PROBE, help="existing frozen probe_T5.npz")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="new output directory; originals are not written")
    ap.add_argument("--T", type=int, default=DEFAULT_T, help="collision-window horizon in frames")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    h = np.load(input_dir / "hidden.npz")["H"].astype(np.float32)
    meta = json.load(open(input_dir / "meta.json"))
    probe_npz = np.load(args.probe)
    probe = {k: probe_npz[k] for k in probe_npz.files}
    if len(meta) != len(h):
        raise ValueError(f"hidden/meta row mismatch: H={len(h)} meta={len(meta)}")
    required_probe = {"mu_pca", "V_pca", "mu_lr", "sd_lr", "w_lr"}
    missing = required_probe - set(probe)
    if missing:
        raise ValueError(f"probe missing keys: {sorted(missing)}")
    if h.shape[1] != len(probe["mu_pca"]):
        raise ValueError(f"hidden dim {h.shape[1]} != probe dim {len(probe['mu_pca'])}")

    cond = np.asarray([m["cond"] for m in meta])
    crashed = np.asarray([bool(m["crashed_episode"]) for m in meta])
    stc = np.asarray([int(m["steps_to_crash"]) for m in meta])
    source = np.flatnonzero((cond == "wall") & crashed & (stc >= 0) & (stc <= args.T))
    offpath_safe = np.flatnonzero((cond == "offpath") & ~crashed)
    offpath_all = np.flatnonzero(cond == "offpath")
    nowall = np.flatnonzero(cond == "nowall")
    if not len(source) or not len(offpath_safe) or not len(nowall):
        raise ValueError("missing one of onpath collision-window, safe offpath, or nowall rows")

    print(f"loaded H={h.shape}; source collision-window rows={len(source)}; "
          f"safe offpath={len(offpath_safe)}; nowall={len(nowall)}", flush=True)
    ts_summary, ts_details = build_pair_records(meta, h, probe, source, offpath_safe, nowall, "timestep")
    nn_summary, nn_details = build_pair_records(meta, h, probe, source, offpath_safe, nowall, "timestep_eef_action_nn")

    indices, labels, sid, folds, pool_counts = prepare_loso_data(meta, h, args.T)
    feature_kinds = [
        ("timestep", "timestep"),
        ("eef_xyz", "EEF xyz"),
        ("timestep_eef_xyz", "timestep + EEF xyz"),
        ("timestep_eef_xyz_action_magnitude", "timestep + EEF xyz + action magnitude"),
        ("hidden_state", "hidden state"),
    ]
    metrics: dict[str, dict[str, Any]] = {}
    oof_for_plot: dict[str, np.ndarray] = {}
    for kind, _ in feature_kinds:
        print(f"LOSO fitting {kind} ...", flush=True)
        result, oof = fit_loso_feature(h, meta, indices, labels, sid, folds, kind)
        metrics[kind] = result
        oof_for_plot[kind] = oof
        print(f"  ROC-AUC={result['roc_auc']:.4f} AUPRC={result['auprc']:.4f}", flush=True)

    # A compact sensitivity check uses all off-path rows (including the five
    # boundary episodes that crashed). It is not the primary clean control, but
    # makes the treatment of those rows explicit and auditable.
    ts_all_summary, ts_all_details = build_pair_records(meta, h, probe, source, offpath_all, nowall, "timestep")

    result = {
        "analysis": {
            "name": "task_phase_confound",
            "T": int(args.T),
            "read_only_inputs": True,
            "primary_offpath_definition": "cond == offpath and crashed_episode == false",
            "reason_for_excluding_offpath_crashes": "wall-visibility control should not also be a collision control",
            "collision_window_definition": "cond == wall, crashed_episode == true, 0 <= steps_to_crash <= T",
            "loso_group_definition": "five on-path scenario IDs; paired no-wall rows share the held-out ID; off-path controls remain available in each fold",
            "positive_rows_for_matching": int(len(source)),
            "source_row_indices": [int(i) for i in source],
            "condition_counts": {
                str(c): int((cond == c).sum()) for c in sorted(set(cond.tolist()))
            },
        },
        "inputs": {
            "capture_dir": str(input_dir),
            "hidden_file": str(input_dir / "hidden.npz"),
            "meta_file": str(input_dir / "meta.json"),
            "probe_file": str(args.probe),
            "hidden_shape": [int(x) for x in h.shape],
            "hidden_dtype_on_disk": str(np.load(input_dir / "hidden.npz")["H"].dtype),
            "sha256": {
                "hidden.npz": hashlib.sha256((input_dir / "hidden.npz").read_bytes()).hexdigest(),
                "meta.json": hashlib.sha256((input_dir / "meta.json").read_bytes()).hexdigest(),
                "probe_T5.npz": hashlib.sha256(Path(args.probe).read_bytes()).hexdigest(),
            },
        },
        "pool": pool_counts,
        "timestep_matching": {"summary": ts_summary, "details": ts_details},
        "timestep_eef_action_nn_matching": {"summary": nn_summary, "details": nn_details},
        "all_offpath_timestep_sensitivity": {"summary": ts_all_summary, "details": ts_all_details},
        "loso_classification": {
            "pool": "all wall rows + all no-wall rows + safe off-path rows; label is on-path collision within T",
            "evaluated_rows": int(metrics["hidden_state"]["n_evaluated"]),
            "feature_sets": metrics,
        },
        "interpretation": {
            "wall_visibility_confound": "The off-path-safe matched group has a wall in the capture but no collision. If its frozen probe logit stays near no-wall, wall visibility alone is insufficient; if it rises, the probe may be reading visibility.",
            "generic_task_phase_confound": "Timestep-only and EEF/action covariate LOSO scores quantify how much collision-window membership can be predicted from progress, robot pose, and action magnitude without hidden state.",
            "collision_specific_information": "Evidence for collision-specific hidden information requires a high hidden-state LOSO score that remains above the covariate baselines and a positive on-path-minus-control logit gap after both matching audits. This is associational, not causal.",
            "final_test": "Compare hidden_state ROC-AUC/AUPRC with timestep + EEF xyz + action magnitude; the latter is the preregistered observable-confound baseline.",
        },
    }
    json_path = output_dir / "task_phase_confound.json"
    json.dump(clean_json(result), open(json_path, "w"), indent=2, allow_nan=False)
    fig_path = output_dir / "task_phase_confound_summary.png"
    make_figure(fig_path, meta, {"summary": ts_summary, "details": ts_details}, {"summary": nn_summary, "details": nn_details}, metrics)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {fig_path}", flush=True)


if __name__ == "__main__":
    main()
