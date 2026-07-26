#!/usr/bin/env python3
"""Leakage-safe analysis of a P0 capture.

Models are fit on training scenarios, thresholds are selected on calibration
scenarios, and the final numbers are computed once on held-out scenarios.  The
diagnostic cross-validation holds out whole task/scenario groups; frames are never
randomly split.  Scenario-macro AUC and scenario-bootstrap differences are primary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.provenance import write_json_exclusive


def auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    good = np.isfinite(scores)
    scores, labels = scores[good], labels[good]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def group_weights(groups: np.ndarray) -> np.ndarray:
    values, counts = np.unique(groups, return_counts=True)
    inverse = {value: 1.0 / count for value, count in zip(values, counts)}
    weights = np.asarray([inverse[value] for value in groups], dtype=float)
    return weights / weights.mean()


def pca_fit(X: np.ndarray, weights: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    weights = weights / weights.sum()
    mu = np.sum(X * weights[:, None], axis=0)
    centered = (X - mu) * np.sqrt(weights[:, None])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mu, vt[:min(k, len(vt))].T


def logreg_fit(X: np.ndarray, y: np.ndarray, weights: np.ndarray, *, l2: float = 2.0,
               iterations: int = 600, learning_rate: float = 0.3):
    weights = weights / weights.sum()
    mu = np.sum(X * weights[:, None], axis=0)
    var = np.sum((X - mu) ** 2 * weights[:, None], axis=0)
    sd = np.sqrt(var) + 1e-6
    xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    coef = np.zeros(xs.shape[1], dtype=float)
    y = y.astype(float)
    for _ in range(iterations):
        logits = np.clip(xs @ coef, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = xs.T @ ((prob - y) * weights)
        grad[:-1] += l2 * coef[:-1] / max(1, len(y))
        coef -= learning_rate * grad
    return mu, sd, coef


def logreg_score(model, X: np.ndarray) -> np.ndarray:
    mu, sd, coef = model
    return ((X - mu) / sd) @ coef[:-1] + coef[-1]


def fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, pca_k: int | None):
    weights = group_weights(groups)
    if pca_k is None:
        mu_pca, vectors = np.zeros(X.shape[1]), np.eye(X.shape[1])
        z = X
    else:
        mu_pca, vectors = pca_fit(X, weights, pca_k)
        z = (X - mu_pca) @ vectors
    lr = logreg_fit(z, y, weights)
    return mu_pca, vectors, lr


def score(model, X: np.ndarray) -> np.ndarray:
    mu_pca, vectors, lr = model
    return logreg_score(lr, (X - mu_pca) @ vectors)


def select_threshold(scores: np.ndarray, y: np.ndarray, max_fpr: float) -> dict:
    if not y.any() or not (~y).any():
        raise ValueError("calibration split needs both positive and negative frames")
    candidates = np.unique(scores)[::-1]
    choices = []
    for threshold in candidates:
        predicted = scores >= threshold
        fpr = float(predicted[~y].mean())
        tpr = float(predicted[y].mean())
        if fpr <= max_fpr:
            choices.append((tpr, -fpr, threshold))
    if not choices:
        threshold = float(np.nextafter(scores.max(), np.inf))
        return {"threshold": threshold, "tpr": 0.0, "fpr": 0.0}
    tpr, neg_fpr, threshold = max(choices)
    return {"threshold": float(threshold), "tpr": float(tpr), "fpr": float(-neg_fpr)}


def macro_auc(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, dict[str, float]]:
    per_group = {group: auc(scores[groups == group], y[groups == group]) for group in np.unique(groups)}
    valid = [value for value in per_group.values() if np.isfinite(value)]
    return (float(np.mean(valid)) if valid else float("nan")), per_group


def grouped_cv(X, y, groups, pca_k) -> dict:
    predictions = np.full(len(y), np.nan)
    for heldout in np.unique(groups):
        train = groups != heldout
        test = ~train
        if not y[train].any() or not (~y[train]).any():
            continue
        predictions[test] = score(fit(X[train], y[train], groups[train], pca_k), X[test])
    good = np.isfinite(predictions)
    macro, per_group = macro_auc(predictions[good], y[good], groups[good])
    return {"frame_auc": auc(predictions[good], y[good]), "scenario_macro_auc": macro,
            "scenario_auc": per_group}


def bootstrap_difference(hidden_group_auc: dict, baseline_group_auc: dict, seed: int,
                         n_bootstrap: int = 5000) -> dict:
    groups = sorted(set(hidden_group_auc) & set(baseline_group_auc))
    groups = [g for g in groups if np.isfinite(hidden_group_auc[g]) and np.isfinite(baseline_group_auc[g])]
    if not groups:
        return {"mean_difference": None, "ci95": [None, None], "n_scenarios": 0}
    differences = np.asarray([hidden_group_auc[g] - baseline_group_auc[g] for g in groups])
    rng = np.random.default_rng(seed)
    boots = np.mean(rng.choice(differences, size=(n_bootstrap, len(differences)), replace=True), axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n_scenarios": len(groups),
        "unit": "scenario",
    }


def load_frames(root: Path, horizon: int):
    episodes = json.loads((root / "episodes.json").read_text())
    columns: dict[str, list[np.ndarray]] = {
        key: [] for key in ("hidden", "time", "eef_pose", "joint_pose", "action", "task_phase")
    }
    labels, splits, groups, tasks = [], [], [], []
    for episode in episodes:
        arrays = np.load(root / episode["arrays"])
        n = len(arrays["hidden"])
        columns["hidden"].append(arrays["hidden"].astype(np.float32))
        columns["time"].append((np.arange(n, dtype=np.float32) / max(1, episode["max_steps"]))[:, None])
        columns["eef_pose"].append(arrays["eef_pose"].astype(np.float32))
        columns["joint_pose"].append(arrays["joint_qpos"].astype(np.float32))
        columns["action"].append(arrays["action"].astype(np.float32))
        phase = np.eye(3, dtype=np.float32)[arrays["task_phase"].astype(int)]
        columns["task_phase"].append(phase)
        stc = arrays["steps_to_impact"]
        labels.append((stc >= 0) & (stc <= horizon))
        split = episode["split"]
        group = f"{episode['task_suite']}:{episode['task_id']}|{episode['scenario_fingerprint_sha256']}"
        task = f"{episode['task_suite']}:{episode['task_id']}"
        splits.extend([split] * n)
        groups.extend([group] * n)
        tasks.extend([task] * n)
    data = {key: np.concatenate(value) for key, value in columns.items()}
    data["robot_state"] = np.concatenate([
        data["time"], data["eef_pose"], data["joint_pose"], data["task_phase"]
    ], axis=1)
    return data, np.concatenate(labels).astype(bool), np.asarray(splits), np.asarray(groups), np.asarray(tasks)


def save_probe(path: Path, model, threshold: float, feature_name: str) -> None:
    mu_pca, vectors, (mu_lr, sd_lr, coef) = model
    np.savez(
        path, mu_pca=mu_pca.astype(np.float32), V_pca=vectors.astype(np.float32),
        mu_lr=mu_lr.astype(np.float32), sd_lr=sd_lr.astype(np.float32),
        w_lr=coef.astype(np.float32), thr=np.float32(threshold), feature_name=feature_name,
    )


def continuous_corridor_analysis(root: Path) -> dict:
    """Primary corridor model: crash ~ continuous full-arm signed clearance."""
    episodes = json.loads((root / "episodes.json").read_text())
    episodes = [row for row in episodes if row["condition"] != "nowall"]
    if not episodes:
        raise ValueError("corridor analysis has no wall/offpath episodes")
    x = np.asarray([[row["corridor"]["signed_distance_m"]] for row in episodes], dtype=float)
    y = np.asarray([row["crashed"] for row in episodes], dtype=bool)
    split = np.asarray([row["split"] for row in episodes])
    group = np.asarray([
        f"{row['task_suite']}:{row['task_id']}|{row['scenario_fingerprint_sha256']}"
        for row in episodes
    ])
    train, heldout = split == "train", split == "heldout"
    if not y[train].any() or not (~y[train]).any():
        raise ValueError("training corridor episodes need both crash and non-crash outcomes")
    model = logreg_fit(x[train], y[train], group_weights(group[train]))
    heldout_score = logreg_score(model, x[heldout])
    _, _, coefficient = model
    scenario_rows = []
    for scenario in np.unique(group[heldout]):
        mask = heldout & (group == scenario)
        source = next(row for row in episodes if (
            f"{row['task_suite']}:{row['task_id']}|{row['scenario_fingerprint_sha256']}" == scenario
            and row["split"] == "heldout"
        ))
        scenario_rows.append({
            "task_scenario": scenario,
            "signed_distance_m": float(x[mask][0, 0]),
            "predeclared_bin": source["corridor"]["predeclared_bin"],
            "n_rollouts": int(mask.sum()),
            "crash_rate": float(y[mask].mean()),
        })
    return {
        "primary_predictor": "continuous signed distance from obstacle to nominal full-arm swept AABB union",
        "statistical_unit": "scenario (rollouts receive inverse within-scenario weights)",
        "train_standardized_clearance_coefficient": float(coefficient[0]),
        "heldout_episode_auc": auc(heldout_score, y[heldout]),
        "heldout_scenarios": scenario_rows,
        "bins_are_descriptive_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--pca-k", type=int, default=50)
    parser.add_argument("--calibration-max-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    args = parser.parse_args()
    root = Path(args.capture_dir).resolve()
    if not (root / "complete.json").exists():
        raise SystemExit(f"capture is incomplete: {root / 'complete.json'} is missing")
    output = root / "probe_analysis"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    data, y, splits, groups, tasks = load_frames(root, args.horizon)
    train, calibration, heldout = splits == "train", splits == "calibration", splits == "heldout"
    if not all(mask.any() for mask in (train, calibration, heldout)):
        raise ValueError("train/calibration/heldout must all contain frames")

    specifications = {
        "hidden": args.pca_k,
        "time": None,
        "eef_pose": None,
        "joint_pose": None,
        "action": None,
        "task_phase": None,
        "robot_state": None,
    }
    summary = {
        "schema_version": 1,
        "horizon_steps": args.horizon,
        "split_unit": "task/scenario; frames are never randomly split",
        "primary_unit": "scenario",
        "n_frames": {name: int((splits == name).sum()) for name in ("train", "calibration", "heldout")},
        "n_scenarios": {name: int(len(np.unique(groups[splits == name])))
                        for name in ("train", "calibration", "heldout")},
        "tasks": sorted(np.unique(tasks).tolist()),
        "calibration_max_fpr": args.calibration_max_fpr,
        "capture_git_commit": json.loads((root / "run_provenance.json").read_text())[
            "repository"]["git_commit"],
        "models": {},
        "corridor_primary_analysis": continuous_corridor_analysis(root),
    }
    fitted = {}
    for name, pca_k in specifications.items():
        X = data[name]
        model = fit(X[train], y[train], groups[train], pca_k)
        calibration_scores = score(model, X[calibration])
        threshold = select_threshold(calibration_scores, y[calibration], args.calibration_max_fpr)
        risk_thresholds = [
            {"target_calibration_fpr": target, **select_threshold(calibration_scores, y[calibration], target)}
            for target in (0.0, 0.01, 0.05, 0.10, 0.20, 0.40)
        ]
        heldout_scores = score(model, X[heldout])
        macro, per_group = macro_auc(heldout_scores, y[heldout], groups[heldout])
        cv_mask = train | calibration
        summary["models"][name] = {
            "grouped_cv_leave_one_scenario_out": grouped_cv(
                X[cv_mask], y[cv_mask], groups[cv_mask], pca_k),
            "grouped_cv_leave_one_task_out": grouped_cv(
                X[cv_mask], y[cv_mask], tasks[cv_mask], pca_k),
            "calibration": threshold,
            "risk_curve_thresholds": risk_thresholds,
            "heldout_frame_auc": auc(heldout_scores, y[heldout]),
            "heldout_scenario_macro_auc": macro,
            "heldout_scenario_auc": per_group,
            "heldout_tpr": float((heldout_scores[y[heldout]] >= threshold["threshold"]).mean())
                           if y[heldout].any() else None,
            "heldout_fpr": float((heldout_scores[~y[heldout]] >= threshold["threshold"]).mean())
                           if (~y[heldout]).any() else None,
        }
        fitted[name] = (model, threshold["threshold"])
        print(f"{name:12s} heldout scenario-macro AUC={macro:.3f}")

    hidden_auc = summary["models"]["hidden"]["heldout_scenario_auc"]
    comparisons = {}
    for name in specifications:
        if name == "hidden":
            continue
        comparisons[name] = bootstrap_difference(
            hidden_auc, summary["models"][name]["heldout_scenario_auc"], args.bootstrap_seed)
    summary["hidden_minus_baseline"] = comparisons
    best_baseline = {}
    for group in hidden_auc:
        candidates = [summary["models"][name]["heldout_scenario_auc"].get(group)
                      for name in specifications if name != "hidden"]
        candidates = [value for value in candidates if value is not None and np.isfinite(value)]
        if candidates:
            best_baseline[group] = max(candidates)
    summary["hidden_minus_best_preregistered_baseline"] = bootstrap_difference(
        hidden_auc, best_baseline, args.bootstrap_seed)
    best_ci = summary["hidden_minus_best_preregistered_baseline"]["ci95"]
    summary["dissociation_supported"] = best_ci[0] is not None and best_ci[0] > 0
    summary["decision_rule"] = (
        "representation-behavior dissociation is supported only if the scenario-bootstrap "
        "95% CI for hidden AUC minus the best preregistered baseline is above zero; "
        "pairwise comparisons are descriptive"
    )
    output.mkdir()
    save_probe(output / "probe_hidden.npz", *fitted["hidden"], feature_name="hidden")
    save_probe(output / "probe_robot_state.npz", *fitted["robot_state"], feature_name="robot_state")
    write_json_exclusive(output / "summary.json", summary)
    print(f"wrote {output}; dissociation_supported={summary['dissociation_supported']}")


if __name__ == "__main__":
    main()
