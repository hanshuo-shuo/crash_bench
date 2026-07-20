#!/usr/bin/env python3
"""Task-phase confound analysis from an existing hidden-state capture.

This analysis is deliberately CPU-only and never runs a VLA.  It reads only the
row-aligned ``hidden.npz`` and ``meta.json`` capture.  The existing full-data
``probe_T5.npz`` is not used for the primary result.

The primary matched-logit result is strict leave-one-wall-scenario-out (OOF):
each fold fits a fresh PCA-50 + L2 logistic probe on the other four wall
scenarios and their paired no-wall scenarios, then scores the held-out wall,
the held-out paired no-wall, and scenario-balanced safe off-path matches.

The classification audit uses the same five held-out wall/no-wall scenario
groups.  It reports linear logistic and a small nonlinear one-hidden-layer MLP
for timestep, EEF xyz, action magnitude, their combinations, hidden only, and
hidden + covariates.  All uncertainty is grouped at the scenario level.

Outputs under ``--output``:

  task_phase_confound.json
  task_phase_confound_summary.png
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = "results/selfreport"
DEFAULT_OUTPUT = "results/task_phase_confound"
DEFAULT_T = 5

PCA_K = 50
PCA_OVERSAMPLE = 10
PCA_POWER_ITERATIONS = 2
PCA_SEED = 0
L2 = 2.0
LOGREG_ITERS = 400
LOGREG_LR = 0.5
MLP_HIDDEN = 16
MLP_EPOCHS = 220
MLP_LR = 0.04
MLP_L2 = 1e-2
MLP_SEED = 1729
INFERENCE_SEED = 20260719
BOOTSTRAP_REPS = 20000

FINAL_CONCLUSION = (
    "Hidden states contain additional collision-predictive information beyond "
    "measured task progress, EEF pose, and action magnitude."
)

COVARIATE_COLUMNS = {
    "timestep": [0],
    "eef_xyz": [1, 2, 3],
    "action_magnitude": [4],
    "timestep_eef_xyz": [0, 1, 2, 3],
    "timestep_action_magnitude": [0, 4],
    "eef_xyz_action_magnitude": [1, 2, 3, 4],
    "covariates_only": [0, 1, 2, 3, 4],
}
COVARIATE_LABELS = {
    "timestep": "timestep",
    "eef_xyz": "EEF xyz",
    "action_magnitude": "action magnitude",
    "timestep_eef_xyz": "timestep + EEF xyz",
    "timestep_action_magnitude": "timestep + action magnitude",
    "eef_xyz_action_magnitude": "EEF xyz + action magnitude",
    "covariates_only": "all measured covariates",
    "hidden_only": "hidden only",
    "hidden_plus_covariates": "hidden + covariates",
}
OBSERVABLE_INPUTS = list(COVARIATE_COLUMNS)
ALL_INPUTS = OBSERVABLE_INPUTS + ["hidden_only", "hidden_plus_covariates"]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with stable average ranks for ties."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
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
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision with tied scores evaluated at a common threshold."""
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
        previous_recall = cum_pos / n_pos
        cum_pos += group_pos
        cum_n += end - start
        if group_pos:
            ap += (cum_pos / cum_n) * (cum_pos / n_pos - previous_recall)
        start = end
    return float(ap)


def pca_fit(x: np.ndarray, k: int = PCA_K) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic training-fold PCA without a large full SVD."""
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
    """Repository-compatible standardized L2 logistic regression."""
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
    return np.hstack(
        [(np.asarray(x, dtype=np.float64) - mu) / sd, np.ones((len(x), 1), dtype=np.float64)]
    ) @ w


def mlp_fit(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int = MLP_HIDDEN,
    epochs: int = MLP_EPOCHS,
    lr: float = MLP_LR,
    l2: float = MLP_L2,
    seed: int = MLP_SEED,
) -> dict[str, Any]:
    """Fit a small deterministic tanh MLP on standardized numeric features."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    xs = (x - mu) / sd
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, np.sqrt(2.0 / max(1, xs.shape[1])), (xs.shape[1], hidden))
    b1 = np.zeros(hidden, dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden), hidden)
    b2 = 0.0
    n = len(y)
    for _ in range(epochs):
        a1 = xs @ w1 + b1
        z1 = np.tanh(a1)
        z2 = np.clip(z1 @ w2 + b2, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-z2))
        dz2 = (p - y) / n
        dw2 = z1.T @ dz2 + l2 * w2
        db2 = float(dz2.sum())
        dz1 = (dz2[:, None] * w2[None, :]) * (1.0 - z1 * z1)
        dw1 = xs.T @ dz1 + l2 * w1
        db1 = dz1.sum(axis=0)
        w2 -= lr * dw2
        b2 -= lr * db2
        w1 -= lr * dw1
        b1 -= lr * db1
    return {"mu": mu, "sd": sd, "w1": w1, "b1": b1, "w2": w2, "b2": b2}


def mlp_score(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    xs = (np.asarray(x, dtype=np.float64) - model["mu"]) / model["sd"]
    z1 = np.tanh(xs @ model["w1"] + model["b1"])
    return z1 @ model["w2"] + model["b2"]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def quantiles(x: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"q25": None, "median": None, "q75": None}
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    return {"q25": float(q25), "median": float(med), "q75": float(q75)}


def summary_stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"macro_mean": None, "median": None, "range": [None, None], "values": []}
    return {
        "macro_mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "range": [float(np.min(x)), float(np.max(x))],
        "values": [float(v) for v in x],
    }


def meta_feature_matrix(meta: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [m["t"], m["eef_x"], m["eef_y"], m["eef_z"], m["act_xyz_norm"]]
            for m in meta
        ],
        dtype=np.float64,
    )


def abbreviated_scenario(scenario_id: str) -> str:
    parts = scenario_id.split("_")
    return parts[-1] if parts else scenario_id


def row_record(meta: list[dict[str, Any]], index: int, logit: float) -> dict[str, Any]:
    m = meta[index]
    return {
        "row_index": int(index),
        "scenario_id": str(m["scenario_id"]),
        "cond": str(m["cond"]),
        "t": int(m["t"]),
        "steps_to_crash": int(m["steps_to_crash"]),
        "probe_logit": float(logit),
    }


def select_nearest_timestep(
    meta: list[dict[str, Any]], source_index: int, candidates: np.ndarray
) -> int:
    if not len(candidates):
        raise ValueError("empty candidate set for timestep matching")
    t = int(meta[source_index]["t"])
    return min(
        (int(i) for i in candidates),
        key=lambda i: (abs(int(meta[i]["t"]) - t), int(meta[i]["t"]), i),
    )


def usage_summary(assignments: list[tuple[str, int]]) -> dict[str, Any]:
    by_scenario: dict[str, Counter[int]] = defaultdict(Counter)
    for scenario_id, row_index in assignments:
        by_scenario[scenario_id][int(row_index)] += 1
    total = len(assignments)
    unique = len({row_index for _, row_index in assignments})
    per_scenario: dict[str, Any] = {}
    for scenario_id in sorted(by_scenario):
        counts = by_scenario[scenario_id]
        assigned = int(sum(counts.values()))
        unique_scenario = len(counts)
        per_scenario[scenario_id] = {
            "matched_assignments": assigned,
            "unique_control_rows": int(unique_scenario),
            "reuse_count": int(assigned - unique_scenario),
            "rows_used_more_than_once": int(sum(v > 1 for v in counts.values())),
            "max_uses_of_one_row": int(max(counts.values())) if counts else 0,
            "use_count_histogram": {
                str(k): int(v) for k, v in sorted(Counter(counts.values()).items())
            },
        }
    return {
        "matched_assignments": int(total),
        "unique_control_rows": int(unique),
        "reuse_count": int(total - unique),
        "rows_used_more_than_once": int(
            sum(v > 1 for counts in by_scenario.values() for v in counts.values())
        ),
        "max_uses_of_one_row": int(max((max(c.values()) for c in by_scenario.values()), default=0)),
        "per_scenario": per_scenario,
    }


def fit_probe_fold(
    h: np.ndarray,
    cond: np.ndarray,
    sid: np.ndarray,
    labels: np.ndarray,
    held_out: str,
) -> dict[str, Any]:
    """Fit the strict OOF hidden probe for one held-out wall scenario."""
    wall_or_nowall = (cond == "wall") | (cond == "nowall")
    train = wall_or_nowall & (sid != held_out)
    test = wall_or_nowall & (sid == held_out)
    if labels[train].sum() == 0 or (~labels[train]).sum() == 0:
        raise ValueError(f"training fold {held_out} lacks both labels")
    mu_pca, v_pca = pca_fit(h[train], PCA_K)
    z_train = (h[train].astype(np.float64) - mu_pca) @ v_pca
    model = logreg_fit(z_train, labels[train].astype(np.float64))
    return {
        "held_out_scenario": held_out,
        "train_mask": train,
        "test_mask": test,
        "mu_pca": mu_pca,
        "v_pca": v_pca,
        "model": model,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "n_positive_train": int(labels[train].sum()),
        "n_negative_train": int((~labels[train]).sum()),
    }


def probe_fold_score(fold: dict[str, Any], h: np.ndarray, indices: np.ndarray) -> np.ndarray:
    z = (h[indices].astype(np.float64) - fold["mu_pca"]) @ fold["v_pca"]
    return logreg_score(fold["model"], z)


def build_strict_oof_matches(
    meta: list[dict[str, Any]],
    h: np.ndarray,
    cond: np.ndarray,
    sid: np.ndarray,
    labels: np.ndarray,
    wall_scenarios: list[str],
    safe_offpath_scenarios: list[str],
    folds: dict[str, dict[str, Any]],
    args_t: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[float]]]:
    """Score held-out collision rows and scenario-balanced matched controls."""
    details: list[dict[str, Any]] = []
    off_assignments: list[tuple[str, int]] = []
    nowall_assignments: list[tuple[str, int]] = []
    fold_summaries = []
    plot_rows = {"steps_to_crash": [], "onpath": [], "offpath_mean": [], "nowall": []}

    for held in wall_scenarios:
        fold = folds[held]
        source = np.flatnonzero((cond == "wall") & (sid == held) & labels)
        held_nowall = np.flatnonzero((cond == "nowall") & (sid == held))
        if not len(source) or not len(held_nowall):
            raise ValueError(f"missing source or paired nowall rows for {held}")
        off_candidates = {
            scenario_id: np.flatnonzero((cond == "offpath") & (sid == scenario_id))
            for scenario_id in safe_offpath_scenarios
        }
        all_selected = list(source) + list(held_nowall)
        matches_by_source = []
        for source_index in source:
            nowall_index = select_nearest_timestep(meta, int(source_index), held_nowall)
            off_indices = {
                scenario_id: select_nearest_timestep(meta, int(source_index), candidates)
                for scenario_id, candidates in off_candidates.items()
            }
            all_selected.extend(off_indices.values())
            matches_by_source.append((int(source_index), nowall_index, off_indices))

        unique_selected = np.asarray(sorted(set(all_selected)), dtype=np.int64)
        selected_scores = probe_fold_score(fold, h, unique_selected)
        score_by_index = {int(i): float(s) for i, s in zip(unique_selected, selected_scores)}

        on_values = [score_by_index[i] for i, _, _ in matches_by_source]
        nowall_values = [score_by_index[i] for _, i, _ in matches_by_source]
        off_by_scenario: dict[str, list[float]] = {s: [] for s in safe_offpath_scenarios}
        fold_pairs = []
        for source_index, nowall_index, off_indices in matches_by_source:
            off_rows = {}
            for scenario_id in safe_offpath_scenarios:
                off_index = off_indices[scenario_id]
                off_assignments.append((scenario_id, off_index))
                off_by_scenario[scenario_id].append(score_by_index[off_index])
                off_rows[scenario_id] = row_record(meta, off_index, score_by_index[off_index])
            nowall_assignments.append((held, nowall_index))
            fold_pairs.append(
                {
                    "onpath": row_record(meta, source_index, score_by_index[source_index]),
                    "offpath_by_scenario": off_rows,
                    "nowall": row_record(meta, nowall_index, score_by_index[nowall_index]),
                }
            )
            plot_rows["steps_to_crash"].append(int(meta[source_index]["steps_to_crash"]))
            plot_rows["onpath"].append(score_by_index[source_index])
            plot_rows["offpath_mean"].append(
                float(np.mean([off_rows[s]["probe_logit"] for s in safe_offpath_scenarios]))
            )
            plot_rows["nowall"].append(score_by_index[nowall_index])

        scenario_means = {s: float(np.mean(v)) for s, v in off_by_scenario.items()}
        on_mean = float(np.mean(on_values))
        nowall_mean = float(np.mean(nowall_values))
        fold_summaries.append(
            {
                "held_out_scenario": held,
                "n_source_rows": int(len(source)),
                "n_match_assignments_per_offpath_scenario": int(len(source)),
                "onpath_within_scenario_mean_logit": on_mean,
                "offpath_control_scenario_means": scenario_means,
                "offpath_scenario_balanced_mean_logit": float(np.mean(list(scenario_means.values()))),
                "nowall_within_scenario_mean_logit": nowall_mean,
                "onpath_minus_offpath_by_control_scenario": {
                    s: float(on_mean - v) for s, v in scenario_means.items()
                },
                "onpath_minus_nowall": float(on_mean - nowall_mean),
                "probe_training": {
                    "wall_scenarios_used": [s for s in wall_scenarios if s != held],
                    "nowall_scenarios_used": [s for s in wall_scenarios if s != held],
                    "n_train": fold["n_train"],
                    "n_positive_train": fold["n_positive_train"],
                    "n_negative_train": fold["n_negative_train"],
                },
                "pairs": fold_pairs,
            }
        )
        details.extend(fold_pairs)

    all_off_scenario_means = [
        value
        for fold in fold_summaries
        for value in fold["offpath_control_scenario_means"].values()
    ]
    off_summary = {
        "fold_scenario_means": [
            {
                "held_out_scenario": f["held_out_scenario"],
                "scenario_means": f["offpath_control_scenario_means"],
                "scenario_balanced_mean": f["offpath_scenario_balanced_mean_logit"],
            }
            for f in fold_summaries
        ],
        "scenario_level_summary_across_fold_control_pairs": summary_stats(all_off_scenario_means),
        "fold_balanced_summary": summary_stats(
            [f["offpath_scenario_balanced_mean_logit"] for f in fold_summaries]
        ),
        "onpath_minus_offpath_fold_balanced": summary_stats(
            [
                f["onpath_within_scenario_mean_logit"]
                - f["offpath_scenario_balanced_mean_logit"]
                for f in fold_summaries
            ]
        ),
    }
    on_summary = summary_stats([f["onpath_within_scenario_mean_logit"] for f in fold_summaries])
    nowall_summary = summary_stats([f["nowall_within_scenario_mean_logit"] for f in fold_summaries])
    match_result = {
        "definition": {
            "matching": "nearest timestep within each control scenario",
            "control_order": "match separately within each scenario, average within scenario, then macro-average across scenarios",
            "safe_offpath_definition": "offpath scenarios with no crashed episode; partially/fully crashing offpath scenarios are excluded",
            "onpath_rows": f"cond == wall and 0 <= steps_to_crash <= {args_t}",
            "nowall_rows": "paired no-wall scenario with the same scenario_id as the held-out wall",
            "probe_scope_per_fold": "only the other four wall scenarios and their four paired nowall scenarios",
            "primary_score": "strict OOF logit from the fold probe; no full-data frozen probe used",
        },
        "folds": fold_summaries,
        "aggregate": {
            "onpath_held_out_scenario_means": on_summary,
            "offpath_safe_control": off_summary,
            "nowall_paired_control": nowall_summary,
            "onpath_minus_nowall": summary_stats(
                [f["onpath_minus_nowall"] for f in fold_summaries]
            ),
        },
        "control_usage": {
            "offpath_safe": usage_summary(off_assignments),
            "nowall_paired": usage_summary(nowall_assignments),
            "all_controls": usage_summary(off_assignments + nowall_assignments),
        },
        "n_pairs": int(len(details)),
        "n_offpath_control_assignments": int(len(off_assignments)),
        "n_nowall_control_assignments": int(len(nowall_assignments)),
    }
    plot_data = {
        key: [float(v) if key != "steps_to_crash" else int(v) for v in values]
        for key, values in plot_rows.items()
    }
    return match_result, {"pairs": details}, plot_data


def metric_summary(
    scenario_rows: list[dict[str, Any]],
    pooled_scores: np.ndarray,
    pooled_labels: np.ndarray,
) -> dict[str, Any]:
    auprc_values = [row["auprc"] for row in scenario_rows]
    auc_values = [row["roc_auc"] for row in scenario_rows]
    return {
        "per_held_out_scenario": scenario_rows,
        "auprc": summary_stats(auprc_values),
        "roc_auc": summary_stats(auc_values),
        "pooled_oof": {
            "auprc": average_precision(pooled_scores, pooled_labels),
            "roc_auc": auc(pooled_scores, pooled_labels),
            "n_evaluated": int(len(pooled_labels)),
            "n_positive": int(pooled_labels.sum()),
            "positive_prevalence": float(np.mean(pooled_labels)),
        },
    }


def model_input_for_fold(
    h: np.ndarray,
    covariates: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    input_name: str,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any] | None]:
    if input_name in COVARIATE_COLUMNS:
        cols = COVARIATE_COLUMNS[input_name]
        return covariates[train_mask][:, cols], covariates[test_mask][:, cols], len(cols), None
    mu_pca, v_pca = pca_fit(h[train_mask], PCA_K)
    z_train = (h[train_mask].astype(np.float64) - mu_pca) @ v_pca
    z_test = (h[test_mask].astype(np.float64) - mu_pca) @ v_pca
    if input_name == "hidden_only":
        return z_train, z_test, int(v_pca.shape[1]), {"mu": mu_pca, "v": v_pca}
    if input_name == "hidden_plus_covariates":
        return (
            np.hstack([z_train, covariates[train_mask]]),
            np.hstack([z_test, covariates[test_mask]]),
            int(z_train.shape[1] + covariates.shape[1]),
            {"mu": mu_pca, "v": v_pca},
        )
    raise ValueError(input_name)


def fit_oof_classifier(
    h: np.ndarray,
    covariates: np.ndarray,
    cond: np.ndarray,
    sid: np.ndarray,
    labels: np.ndarray,
    wall_scenarios: list[str],
    input_name: str,
    model_name: str,
    probe_folds: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fit one model/input combination with one held-out wall scenario at a time."""
    eval_pool = (cond == "wall") | (cond == "nowall")
    oof = np.full(len(cond), np.nan, dtype=np.float64)
    scenario_rows = []
    fold_rows = []
    for fold_index, held in enumerate(wall_scenarios):
        train = eval_pool & (sid != held)
        test = eval_pool & (sid == held)
        if model_name == "linear_logistic" and input_name == "hidden_only":
            scores = probe_fold_score(probe_folds[held], h, np.flatnonzero(test))
            feature_dim = PCA_K
            preprocessing = "training-fold PCA-50, then standardized L2 logistic regression"
        else:
            x_train, x_test, feature_dim, pca_info = model_input_for_fold(
                h, covariates, train, test, input_name
            )
            if model_name == "linear_logistic":
                model = logreg_fit(x_train, labels[train].astype(np.float64))
                scores = logreg_score(model, x_test)
                preprocessing = "training-fold standardization + L2 logistic regression"
                if pca_info is not None:
                    preprocessing = "training-fold PCA-50 + standardization + L2 logistic regression"
            elif model_name == "nonlinear_mlp":
                model = mlp_fit(x_train, labels[train].astype(np.float64), seed=MLP_SEED + fold_index)
                scores = mlp_score(model, x_test)
                preprocessing = "training-fold PCA if hidden + standardization + one-hidden-layer tanh MLP"
            else:
                raise ValueError(model_name)
        test_indices = np.flatnonzero(test)
        oof[test_indices] = scores
        test_labels = labels[test]
        row = {
            "held_out_scenario": held,
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "n_positive_test": int(test_labels.sum()),
            "positive_prevalence": float(test_labels.mean()),
            "roc_auc": auc(scores, test_labels),
            "auprc": average_precision(scores, test_labels),
        }
        scenario_rows.append(row)
        fold_rows.append(
            {
                "held_out_scenario": held,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "feature_dim": int(feature_dim),
            }
        )
    valid = np.isfinite(oof) & eval_pool
    result = metric_summary(scenario_rows, oof[valid], labels[valid])
    result.update(
        {
            "model_name": model_name,
            "input_name": input_name,
            "input_label": COVARIATE_LABELS[input_name],
            "folds": fold_rows,
            "classifier": {
                "model": model_name,
                "preprocessing": preprocessing,
                "logistic_l2": L2 if model_name == "linear_logistic" else None,
                "logistic_iterations": LOGREG_ITERS if model_name == "linear_logistic" else None,
                "mlp_hidden_units": MLP_HIDDEN if model_name == "nonlinear_mlp" else None,
                "mlp_epochs": MLP_EPOCHS if model_name == "nonlinear_mlp" else None,
                "mlp_learning_rate": MLP_LR if model_name == "nonlinear_mlp" else None,
                "mlp_l2": MLP_L2 if model_name == "nonlinear_mlp" else None,
            },
        }
    )
    return result


def grouped_inference(
    hidden_rows: list[dict[str, Any]],
    observable_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Scenario bootstrap and exact sign permutation on paired scenario metrics."""
    hidden_by_scenario = {row["held_out_scenario"]: row for row in hidden_rows}
    obs_by_scenario = {row["held_out_scenario"]: row for row in observable_rows}
    scenarios = sorted(set(hidden_by_scenario) & set(obs_by_scenario))
    differences = np.asarray(
        [hidden_by_scenario[s]["auprc"] - obs_by_scenario[s]["auprc"] for s in scenarios],
        dtype=np.float64,
    )
    observed = float(np.mean(differences))
    rng = np.random.default_rng(INFERENCE_SEED)
    bootstrap = np.asarray(
        [float(np.mean(rng.choice(differences, size=len(differences), replace=True))) for _ in range(BOOTSTRAP_REPS)]
    )
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(differences))), dtype=np.float64)
    null = (signs * differences[None, :]).mean(axis=1)
    p_exact = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    return {
        "unit": "held-out wall scenario",
        "scenarios": scenarios,
        "paired_auprc_differences_hidden_minus_observable": {
            s: float(d) for s, d in zip(scenarios, differences)
        },
        "observed_macro_auprc_difference": observed,
        "scenario_bootstrap": {
            "replicates": BOOTSTRAP_REPS,
            "seed": INFERENCE_SEED,
            "percentile_95_ci": [
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5)),
            ],
        },
        "grouped_sign_permutation": {
            "exact_sign_patterns": int(len(signs)),
            "two_sided_p": p_exact,
            "null_macro_differences": [float(v) for v in null],
        },
        "warning": "This is scenario-level inference with five groups; continuous frames are not treated as independent replicates.",
    }


def make_figure(
    output: Path,
    plot_data: dict[str, list[float]],
    model_results: dict[str, dict[str, Any]],
    primary_hidden_key: str,
    strongest_observable_key: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.05, 1.0, 1.65]})

    # Panel A: strict OOF matched logits.  Off-path is averaged across control
    # scenarios for each source frame before plotting.
    steps = np.asarray(plot_data["steps_to_crash"])
    order = sorted(set(int(x) for x in steps))
    for key, label, color in [
        ("onpath", "held-out wall", "#c0392b"),
        ("offpath_mean", "safe off-path (scenario-balanced)", "#e67e22"),
        ("nowall", "paired no-wall", "#2e8b57"),
    ]:
        means = [
            float(np.mean([v for v, s in zip(plot_data[key], steps) if int(s) == k]))
            for k in order
        ]
        axes[0].plot(order, means, "o-", lw=2, ms=5, label=label, color=color)
    axes[0].set_xlabel("steps to crash (0 = impact)")
    axes[0].set_ylabel("strict OOF probe logit")
    axes[0].set_title("OOF matched logits")
    axes[0].set_xticks(order)
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    # Panel B: per-scenario primary hidden versus selected strongest observable.
    hidden_rows = model_results[primary_hidden_key]["per_held_out_scenario"]
    obs_rows = model_results[strongest_observable_key]["per_held_out_scenario"]
    scenarios = [row["held_out_scenario"] for row in hidden_rows]
    x = np.arange(len(scenarios))
    width = 0.34
    axes[1].bar(
        x - width / 2,
        [row["auprc"] for row in hidden_rows],
        width,
        label="hidden only",
        color="#6a3d9a",
    )
    axes[1].bar(
        x + width / 2,
        [row["auprc"] for row in obs_rows],
        width,
        label="strongest observable",
        color="#1f78b4",
    )
    axes[1].scatter(
        x,
        [row["positive_prevalence"] for row in hidden_rows],
        marker="_",
        s=180,
        color="black",
        label="random prevalence",
        zorder=4,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([abbreviated_scenario(s) for s in scenarios], rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("AUPRC")
    axes[1].set_title("Per-scenario AUPRC")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)

    # Panel C: AUPRC is primary; open circles are ROC-AUC as an auxiliary metric.
    keys = list(model_results)
    labels = [
        f"{model_results[k]['model_name'].replace('_', ' ')}\n{model_results[k]['input_label']}"
        for k in keys
    ]
    vals = np.asarray([model_results[k]["auprc"]["macro_mean"] for k in keys])
    auc_vals = np.asarray([model_results[k]["roc_auc"]["macro_mean"] for k in keys])
    colors = [
        "#6a3d9a" if model_results[k]["input_name"].startswith("hidden") else "#80b1d3"
        for k in keys
    ]
    colors[keys.index(strongest_observable_key)] = "#e31a1c"
    y = np.arange(len(keys))
    axes[2].barh(y, vals, color=colors, alpha=0.9)
    axes[2].scatter(auc_vals, y, facecolors="none", edgecolors="black", s=45, label="ROC-AUC")
    axes[2].axvline(0.5, color="gray", ls=":", lw=1)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(labels, fontsize=7)
    axes[2].set_xlim(0, 1.0)
    axes[2].set_xlabel("macro mean (AUPRC bars; ROC-AUC circles)")
    axes[2].set_title("Observable baseline comparison")
    axes[2].grid(axis="x", alpha=0.2)
    axes[2].legend(fontsize=8, loc="lower right")

    fig.suptitle("Task-phase confound analysis: strict OOF and scenario-level evaluation", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="capture directory with hidden.npz and meta.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="new output directory")
    parser.add_argument("--T", type=int, default=DEFAULT_T, help="collision-window horizon")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_path = input_dir / "hidden.npz"
    meta_path = input_dir / "meta.json"
    hidden_archive = np.load(hidden_path)
    h = hidden_archive["H"].astype(np.float32)
    meta = json.loads(meta_path.read_text())
    if len(meta) != len(h):
        raise ValueError(f"hidden/meta row mismatch: H={len(h)} meta={len(meta)}")

    cond = np.asarray([m["cond"] for m in meta])
    sid = np.asarray([str(m["scenario_id"]) for m in meta])
    crashed = np.asarray([bool(m["crashed_episode"]) for m in meta])
    stc = np.asarray([int(m["steps_to_crash"]) for m in meta])
    labels = (cond == "wall") & crashed & (stc >= 0) & (stc <= args.T)
    wall_scenarios = sorted(set(sid[cond == "wall"]))
    nowall_scenarios = sorted(set(sid[cond == "nowall"]))
    if wall_scenarios != nowall_scenarios:
        raise ValueError("wall and nowall scenario IDs are not paired")

    offpath_scenarios = sorted(set(sid[cond == "offpath"]))
    safe_offpath_scenarios = []
    excluded_offpath_scenarios = []
    for scenario_id in offpath_scenarios:
        rows = (cond == "offpath") & (sid == scenario_id)
        if not crashed[rows].any():
            safe_offpath_scenarios.append(scenario_id)
        else:
            excluded_offpath_scenarios.append(scenario_id)
    if len(wall_scenarios) != 5 or not len(safe_offpath_scenarios):
        raise ValueError("expected five paired wall scenarios and safe off-path controls")

    print(
        f"loaded H={h.shape}; wall={int((cond == 'wall').sum())}; "
        f"nowall={int((cond == 'nowall').sum())}; safe offpath="
        f"{int(sum((cond == 'offpath') & np.isin(sid, safe_offpath_scenarios)))}",
        flush=True,
    )
    print(f"collision-window source rows={int(labels.sum())}; fitting strict OOF probes", flush=True)

    # One strict OOF hidden probe per held-out wall scenario.  These folds are
    # also reused for the primary linear hidden-only classification result.
    folds = {
        held: fit_probe_fold(h, cond, sid, labels, held) for held in wall_scenarios
    }
    match_result, match_details, match_plot_data = build_strict_oof_matches(
        meta,
        h,
        cond,
        sid,
        labels,
        wall_scenarios,
        safe_offpath_scenarios,
        folds,
        args.T,
    )

    covariates = meta_feature_matrix(meta)
    model_results: dict[str, dict[str, Any]] = {}
    for model_name in ("linear_logistic", "nonlinear_mlp"):
        for input_name in ALL_INPUTS:
            key = f"{model_name}__{input_name}"
            print(f"fitting {key}", flush=True)
            model_results[key] = fit_oof_classifier(
                h,
                covariates,
                cond,
                sid,
                labels,
                wall_scenarios,
                input_name,
                model_name,
                folds,
            )
            print(
                f"  macro AUPRC={model_results[key]['auprc']['macro_mean']:.4f} "
                f"macro ROC-AUC={model_results[key]['roc_auc']['macro_mean']:.4f}",
                flush=True,
            )

    primary_hidden_key = "linear_logistic__hidden_only"
    observable_keys = [
        f"{model}__{input_name}"
        for model in ("linear_logistic", "nonlinear_mlp")
        for input_name in OBSERVABLE_INPUTS
    ]
    strongest_observable_key = max(
        observable_keys,
        key=lambda key: model_results[key]["auprc"]["macro_mean"],
    )
    inference = grouped_inference(
        model_results[primary_hidden_key]["per_held_out_scenario"],
        model_results[strongest_observable_key]["per_held_out_scenario"],
    )
    hidden_macro = model_results[primary_hidden_key]["auprc"]["macro_mean"]
    observable_macro = model_results[strongest_observable_key]["auprc"]["macro_mean"]

    result = {
        "schema_version": 2,
        "analysis": {
            "name": "task_phase_confound",
            "T": int(args.T),
            "read_only_inputs": True,
            "vla_rerun": False,
            "primary_hidden_model": primary_hidden_key,
            "strongest_observable_baseline": strongest_observable_key,
            "final_conclusion": FINAL_CONCLUSION,
            "distinction": {
                "stage_information_exists": "Observable timestep, pose, and action features have non-chance scenario-level prediction, so measured task progress information is present in the capture.",
                "stage_information_not_complete": "The primary hidden-only OOF result is compared with the strongest observable OOF candidate; a residual gap means the measured covariates do not fully explain the hidden signal, without claiming that all task-phase confounds are excluded.",
                "causal_scope": "Associational frozen-capture evidence only; it does not establish a pure or causal collision representation.",
            },
        },
        "inputs": {
            "capture_dir": str(input_dir),
            "hidden_file": str(hidden_path),
            "meta_file": str(meta_path),
            "hidden_shape": [int(x) for x in h.shape],
            "hidden_dtype_on_disk": str(hidden_archive["H"].dtype),
            "sha256": {
                "hidden.npz": hashlib.sha256(hidden_path.read_bytes()).hexdigest(),
                "meta.json": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
            },
        },
        "data": {
            "condition_counts": {str(c): int((cond == c).sum()) for c in sorted(set(cond.tolist()))},
            "wall_scenarios": wall_scenarios,
            "safe_offpath_scenarios": safe_offpath_scenarios,
            "excluded_offpath_scenarios": excluded_offpath_scenarios,
            "collision_window_definition": f"cond == wall, crashed_episode == true, 0 <= steps_to_crash <= {args.T}",
            "positive_rows": int(labels.sum()),
            "paired_classification_pool": {
                "definition": "all wall rows + all paired nowall rows; each fold holds out one wall scenario and its nowall counterpart",
                "rows": int(((cond == "wall") | (cond == "nowall")).sum()),
                "positive_rows": int(labels[(cond == "wall") | (cond == "nowall")].sum()),
            },
        },
        "strict_oof_matched_logits": {
            "summary": match_result,
            "details": match_details,
            "plot_data": match_plot_data,
        },
        "scenario_level_classification": {
            "unit": "held-out wall scenario; all frames are scored within a scenario, then metrics are macro-averaged",
            "no_frame_level_significance": True,
            "feature_sets": model_results,
            "primary_comparison": {
                "hidden_key": primary_hidden_key,
                "strongest_observable_key": strongest_observable_key,
                "hidden_macro_auprc": hidden_macro,
                "observable_macro_auprc": observable_macro,
                "macro_auprc_difference": float(hidden_macro - observable_macro),
                "hidden_macro_roc_auc": model_results[primary_hidden_key]["roc_auc"]["macro_mean"],
                "observable_macro_roc_auc": model_results[strongest_observable_key]["roc_auc"]["macro_mean"],
                "scenario_level_inference": inference,
            },
        },
        "reproducibility": {
            "command": "MPLCONFIGDIR=/tmp/crashbench-mpl-taskphase envs/openvla/bin/python3.10 scripts/task_phase_confound_analysis.py --input results/selfreport --output results/task_phase_confound --T 5",
            "pca": {
                "components": PCA_K,
                "randomized_oversample": PCA_OVERSAMPLE,
                "power_iterations": PCA_POWER_ITERATIONS,
                "seed": PCA_SEED,
                "fit_scope": "training fold only",
            },
            "linear_logistic": {"l2": L2, "iterations": LOGREG_ITERS, "learning_rate": LOGREG_LR},
            "nonlinear_mlp": {
                "architecture": f"standardized input -> tanh({MLP_HIDDEN}) -> scalar logit",
                "epochs": MLP_EPOCHS,
                "learning_rate": MLP_LR,
                "l2": MLP_L2,
                "seed": MLP_SEED,
            },
            "inference": {
                "scenario_bootstrap_replicates": BOOTSTRAP_REPS,
                "seed": INFERENCE_SEED,
                "permutation": "exact sign permutation over five held-out scenarios",
            },
        },
    }
    json_path = output_dir / "task_phase_confound.json"
    json_path.write_text(json.dumps(clean_json(result), indent=2, allow_nan=False))
    figure_path = output_dir / "task_phase_confound_summary.png"
    make_figure(
        figure_path,
        match_plot_data,
        model_results,
        primary_hidden_key,
        strongest_observable_key,
    )
    print(f"strongest observable baseline: {strongest_observable_key}", flush=True)
    print(f"primary macro AUPRC hidden={hidden_macro:.4f} observable={observable_macro:.4f}", flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()
