#!/usr/bin/env python3
"""Fixed, source-disjoint diagnostic of Detour completion versus Base risk.

All numerical research work runs in Slurm. No simulator or model inference is
imported. The only model family is a fixed regularized linear logistic probe.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from crashbench.counterfactual_router import OPTIONS, validate_decision_rows
from scripts.train_minimal_counterfactual_router import fit_frame_pca, load_capture


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dump_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def source_weights(sources):
    counts = Counter(map(str, sources))
    w = np.array([1.0 / counts[str(s)] for s in sources], dtype=float)
    return w / w.sum()


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -np.asarray(x)))


def fit_logistic(x, y, sources, l2=0.01):
    """Source-weighted logistic loss + L2/2; Newton steps with line search."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = source_weights(sources)
    if len(x) != len(y) or not len(x) or not np.isfinite(x).all():
        raise ValueError("invalid training data")
    mean = np.sum(w[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(w[:, None] * (x - mean) ** 2, axis=0)) + 1e-6
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    prevalence = float(w @ y)
    if prevalence in (0.0, 1.0):
        return {"mean": mean, "scale": scale, "coef": np.zeros(x.shape[1] + 1),
                "constant": np.array(prevalence)}, {"converged": True, "single_class": True, "iterations": 0}
    penalty = np.full(design.shape[1], float(l2))
    penalty[-1] = 0.0
    coef = np.zeros(design.shape[1])
    coef[-1] = np.log(prevalence / (1 - prevalence))

    def objective(beta):
        z = design @ beta
        return float(w @ (np.logaddexp(0, z) - y * z) + 0.5 * np.sum(penalty * beta ** 2))

    converged = False
    for iteration in range(100):
        p = sigmoid(design @ coef)
        gradient = design.T @ (w * (p - y)) + penalty * coef
        if np.max(np.abs(gradient)) < 1e-8:
            converged = True
            break
        hessian = design.T @ ((w * p * (1 - p))[:, None] * design)
        hessian += np.diag(penalty + 1e-12)
        step = np.linalg.solve(hessian, gradient)
        old_loss = objective(coef)
        rate = 1.0
        for _ in range(40):
            candidate = coef - rate * step
            if objective(candidate) <= old_loss - 1e-4 * rate * float(gradient @ step):
                coef = candidate
                break
            rate *= 0.5
        else:
            raise RuntimeError("logistic line search failed")
    if not converged:
        raise RuntimeError("logistic probe did not converge")
    return {"mean": mean, "scale": scale, "coef": coef}, {
        "converged": True, "single_class": False, "iterations": iteration,
        "training_source_weighted_prevalence": prevalence, "loss": objective(coef),
    }


def predict(model, x):
    if "constant" in model:
        return np.full(len(x), float(model["constant"]))
    design = np.column_stack(((x - model["mean"]) / model["scale"], np.ones(len(x))))
    return sigmoid(design @ model["coef"])


def binary_metrics(y, p, weights=None):
    """Weighted AUROC/AP with exact tie handling; missing classes are explicit."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    w = np.ones(len(y), float) if weights is None else np.asarray(weights, float)
    keep = w > 0
    y, p, w = y[keep], p[keep], w[keep]
    if not len(y):
        return {k: None for k in ("auc", "ap", "brier", "balanced_accuracy", "prevalence")}
    w = w / w.sum()
    positive, negative = float(w @ y), float(w @ (1 - y))
    order = np.argsort(p, kind="stable")
    scores, ys, ws = p[order], y[order], w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    pos = np.add.reduceat(ws * ys, starts)
    neg = np.add.reduceat(ws * (1 - ys), starts)
    auc = None
    if positive > 0 and negative > 0:
        auc = float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (positive * negative))
    ap = None
    if positive > 0:
        tp, total = np.cumsum(pos[::-1]), np.cumsum((pos + neg)[::-1])
        ap = float(np.sum(pos[::-1] / positive * tp / total))
    ba = None
    if positive > 0 and negative > 0:
        ba = float(0.5 * ((w * y) @ (p >= 0.5) / positive + (w * (1 - y)) @ (p < 0.5) / negative))
    return {"auc": auc, "ap": ap, "brier": float(w @ (p - y) ** 2),
            "balanced_accuracy": ba, "prevalence": positive}


def fold_indices(sources, conditions, splits, regime):
    eligible = conditions == "glass" if regime == "glass_loso" else np.ones(len(sources), bool)
    if regime == "historical_train_to_development":
        folds = [("historical", np.flatnonzero(splits == "train"), np.flatnonzero(splits == "development"))]
    else:
        folds = [(str(s), np.flatnonzero(eligible & (sources != s)),
                  np.flatnonzero(eligible & (sources == s))) for s in sorted(set(sources[eligible]))]
    for name, train, test in folds:
        if not len(train) or not len(test) or set(sources[train]) & set(sources[test]):
            raise ValueError("empty or source-leaking fold")
        yield name, train, test


def validate_capture(config):
    root = Path(config["capture"])
    for name, expected in config["input_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"sealed input hash mismatch: {name}")
    metadata = json.loads((root / "decision_metadata.json").read_text())
    if set(r["split"] for r in metadata) != {"train", "calibration", "development"}:
        raise ValueError("only the historical exposed splits are allowed")
    rows = [json.loads(line) for line in (root / "option_rollouts.jsonl").read_text().splitlines() if line.strip()]
    validation = validate_decision_rows(rows)
    if validation["decision_states"] != 273 or validation["source_states"] != 20 or len(rows) != 819:
        raise ValueError("not the authorized 273-decision corpus")
    by_id = {r["decision_id"]: r for r in metadata}
    if len(by_id) != 273:
        raise ValueError("duplicate decision ID")
    seen = set()
    for row in rows:
        pair = (row["decision_id"], row["option"])
        if pair in seen:
            raise ValueError("duplicate option outcome")
        seen.add(pair)
        for key in ("feature_index", "source_state_sha256", "split", "condition", "horizon_actions", "placement_key"):
            if row[key] != by_id[row["decision_id"]][key]:
                raise ValueError(f"metadata/outcome mismatch: {key}")
    data = load_capture(root, catastrophe_cost=1.0)
    for value in data["arrays"].values():
        if not np.isfinite(value).all():
            raise ValueError("nonfinite features")
    if data["arrays"]["hidden"].shape != (273, 8, 4096) or not np.all(data["arrays"]["history_mask"][:, -1] == 1):
        raise ValueError("unexpected current hidden state or invalid last frame")
    return data, validation


def fit_suite(data, config, output):
    n = len(data["sources"])
    hidden = data["arrays"]["hidden"][:, -1:, :]
    robot = np.concatenate([data["arrays"][k][:, -1] for k in ("robot_state", "nominal_action")], axis=1)
    # Category vocabulary is predeclared; categories are privileged diagnostics.
    metadata = np.column_stack([data["conditions"] == c for c in ("glass", "offpath", "noglass")]
                                + [data["horizons"] == h for h in (5, 10, 20, 30, 40)]).astype(float)
    outcomes = data["outcomes"]
    targets = {"base_crash": (outcomes[:, OPTIONS.index("base_continue")] == "catastrophe").astype(int),
               "detour_success": (outcomes[:, OPTIONS.index("detour_complete")] == "task_success").astype(int),
               "base_success": (outcomes[:, OPTIONS.index("base_continue")] == "task_success").astype(int)}
    predictions, fold_log, parameters = [], [], {}
    for regime in config["regimes"]:
        for fold_number, (fold, train, test) in enumerate(fold_indices(data["sources"], data["conditions"], data["splits"], regime)):
            train_mask = np.zeros(n, bool)
            train_mask[train] = True
            pca_mean, components = fit_frame_pca(hidden, np.ones((n, 1)), train_mask,
                                                config["pca_components"], seed=config["seed"])
            projected = (hidden[:, 0].astype(float) - pca_mean) @ components
            features = {"hidden": projected, "robot_action": robot,
                        "hidden_robot_action": np.column_stack((projected, robot)),
                        "oracle_metadata": metadata, "prior": np.empty((n, 0))}
            prefix = f"{regime}_{fold_number}"
            parameters[prefix + "_pca_mean"] = pca_mean
            parameters[prefix + "_pca_components"] = components
            for feature in config["feature_sets"]:
                for target in config["targets"]:
                    model, diagnostics = fit_logistic(features[feature][train], targets[target][train],
                                                      data["sources"][train], config["l2"])
                    for name, array in model.items():
                        parameters[f"{prefix}_{feature}_{target}_{name}"] = array
                    scores = predict(model, features[feature][test])
                    fold_log.append({"regime": regime, "fold": fold, "parameter_prefix": prefix,
                        "feature": feature, "target": target,
                        "train_sources": sorted(set(data["sources"][train].tolist())),
                        "test_sources": sorted(set(data["sources"][test].tolist())),
                        "train_n": len(train), "test_n": len(test),
                        "train_positive_n": int(targets[target][train].sum()),
                        "train_positive_sources": len(set(data["sources"][train][targets[target][train] == 1])),
                        **diagnostics})
                    for index, score in zip(test, scores):
                        row = data["metadata"][index]
                        predictions.append({"regime": regime, "feature": feature, "target": target,
                            "decision_id": row["decision_id"], "source": row["source_state_sha256"],
                            "split": row["split"], "condition": row["condition"], "horizon": row["horizon_actions"],
                            "placement": row["placement_key"], "fold": fold,
                            "y": int(targets[target][index]), "score": float(score),
                            **{name: int(y[index]) for name, y in targets.items()}})
            print(f"completed {regime} fold {fold_number + 1}: train={len(train)} test={len(test)}", flush=True)
    dump_json(output / "folds.json", fold_log)
    np.savez_compressed(output / "models.npz", **parameters)
    # A fixed comparison asks whether inverse crash score already ranks recovery.
    inverse = [dict(row, feature="inverse_risk_hidden", target="detour_success",
                    y=row["detour_success"], score=1 - row["score"])
               for row in predictions if row["feature"] == "hidden" and row["target"] == "base_crash"]
    return predictions + inverse


def scope_mask(rows, scope):
    if scope == "all":
        return np.ones(len(rows), bool)
    if scope == "glass":
        return np.array([r["condition"] == "glass" for r in rows])
    if scope == "controls":
        return np.array([r["condition"] != "glass" for r in rows])
    if scope == "base_failure":
        return np.array([r["base_success"] == 0 for r in rows])
    if scope == "glass_base_failure":
        return scope_mask(rows, "glass") & scope_mask(rows, "base_failure")
    if scope == "glass_base_crash":
        return scope_mask(rows, "glass") & np.array([r["base_crash"] == 1 for r in rows])
    raise ValueError(scope)


SCOPES = ("all", "glass", "controls", "base_failure", "glass_base_failure", "glass_base_crash")


def group_predictions(predictions):
    groups = {}
    for row in predictions:
        key = (row["regime"], row["feature"], row["target"])
        groups.setdefault(key, []).append(row)
    return {key: sorted(rows, key=lambda r: r["decision_id"]) for key, rows in groups.items()}


def summarize(predictions, output):
    metrics, calibration, low_scores, source_table = [], [], [], []
    for key, rows in group_predictions(predictions).items():
        identity = dict(zip(("regime", "feature", "target"), key))
        selections = {scope: scope_mask(rows, scope) for scope in SCOPES}
        for horizon in (5, 10, 20, 30, 40):
            h = np.array([r["horizon"] == horizon for r in rows])
            selections[f"T-{horizon}"] = h
            selections[f"glass_T-{horizon}"] = h & scope_mask(rows, "glass")
        for scope, mask in selections.items():
            local = [r for r, keep in zip(rows, mask) if keep]
            if not local:
                continue
            y = np.array([r["y"] for r in local])
            p = np.array([r["score"] for r in local])
            sources = np.array([r["source"] for r in local])
            within = []
            for source in sorted(set(sources)):
                sm = sources == source
                result = binary_metrics(y[sm], p[sm])
                if result["auc"] is not None:
                    within.append(result["auc"])
                if scope in ("all", "glass"):
                    source_table.append({**identity, "scope": scope, "source": source,
                                         "n": int(sm.sum()), "positive_n": int(y[sm].sum()),
                                         "mean_score": float(p[sm].mean()), **result})
            for weighting, weights in (("decision", np.ones(len(y))), ("source", source_weights(sources))):
                metrics.append({**identity, "scope": scope, "weighting": weighting,
                    "n": len(y), "sources": len(set(sources)), "positive_n": int(y.sum()),
                    "positive_sources": len(set(sources[y == 1])), "negative_sources": len(set(sources[y == 0])),
                    "within_source_auc": float(np.mean(within)) if within else None,
                    "within_source_identifiable_n": len(within), **binary_metrics(y, p, weights)})
            if scope in ("all", "glass"):
                w = source_weights(sources)
                for i in range(5):
                    selected = (p >= i / 5) & (p < (i + 1) / 5 if i < 4 else p <= 1)
                    calibration.append({**identity, "scope": scope, "bin_lower": i / 5,
                        "bin_upper": (i + 1) / 5, "n": int(selected.sum()),
                        "sources": len(set(sources[selected])),
                        "mean_score": float(np.average(p[selected], weights=w[selected])) if selected.any() else None,
                        "observed_rate": float(np.average(y[selected], weights=w[selected])) if selected.any() else None})
            if key[1:3] == ("hidden", "detour_success") and scope in SCOPES:
                for threshold in (0.1, 0.2, 0.5):
                    selected = [r for r in local if r["score"] < threshold]
                    low_scores.append({**identity, "scope": scope, "threshold": threshold,
                        "total_n": len(local), "flagged_n": len(selected),
                        "flagged_sources": len({r["source"] for r in selected}),
                        "detour_success_n": sum(r["detour_success"] for r in selected),
                        "base_success_n": sum(r["base_success"] for r in selected),
                        "either_success_n": sum(bool(r["base_success"] or r["detour_success"]) for r in selected)})
    for filename, rows in (("metrics.csv", metrics), ("calibration.csv", calibration),
                           ("low_score_audit.csv", low_scores), ("per_source.csv", source_table)):
        dump_csv(output / filename, rows)
    return metrics


def bootstrap_comparison(rows, other, repeats, seed):
    """Paired source-block resampling; equal mass for each sampled source copy."""
    if [r["decision_id"] for r in rows] != [r["decision_id"] for r in other]:
        raise ValueError("unpaired bootstrap")
    sources = np.array([r["source"] for r in rows])
    unique, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
    y, p = np.array([r["y"] for r in rows]), np.array([r["score"] for r in rows])
    y2, p2 = np.array([r["y"] for r in other]), np.array([r["score"] for r in other])
    rng = np.random.default_rng(seed)
    values = {f"{prefix}_{m}": [] for prefix in ("left", "right", "difference") for m in ("auc", "brier")}
    for _ in range(repeats):
        multiplicity = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        w = multiplicity[inverse] / counts[inverse]
        left, right = binary_metrics(y, p, w), binary_metrics(y2, p2, w)
        for m in ("auc", "brier"):
            for prefix, result in (("left", left), ("right", right)):
                if result[m] is not None:
                    values[f"{prefix}_{m}"].append(result[m])
            if left[m] is not None and right[m] is not None:
                values[f"difference_{m}"].append(left[m] - right[m])
    result = {}
    left, right = binary_metrics(y, p, source_weights(sources)), binary_metrics(y2, p2, source_weights(sources))
    for key, samples in values.items():
        prefix, metric = key.rsplit("_", 1)
        estimate = left[metric] if prefix == "left" else right[metric]
        if prefix == "difference":
            estimate = left[metric] - right[metric] if left[metric] is not None and right[metric] is not None else None
        result[key] = {"estimate": estimate, "valid_repeats": len(samples),
                       "ci95": np.quantile(samples, [0.025, 0.975]).tolist() if samples else None}
    return result


def inference(predictions, config, output):
    groups = group_predictions(predictions)
    results = []
    for regime in ("pooled_loso", "glass_loso"):
        for scope in (("all", "glass", "glass_base_failure") if regime == "pooled_loso" else ("glass",)):
            left_all = groups[regime, "hidden", "detour_success"]
            mask = scope_mask(left_all, scope)
            left = [r for r, keep in zip(left_all, mask) if keep]
            comparators = [(f, "detour_success") for f in ("prior", "robot_action", "inverse_risk_hidden")]
            comparators.append(("hidden", "base_crash"))
            for feature, target in comparators:
                right_all = groups[regime, feature, target]
                right = [r for r, keep in zip(right_all, mask) if keep]
                results.append({"regime": regime, "scope": scope, "left": "hidden:detour_success",
                                "right": feature + ":" + target,
                                "interpretation": "different-target descriptive contrast" if target == "base_crash" else "same-target comparison",
                                **bootstrap_comparison(left, right, config["bootstrap_repeats"], config["seed"])})
            print(f"bootstrap completed: {regime} {scope}", flush=True)
    dump_json(output / "bootstrap.json", results)


def plot_results(metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    colors = ["#2878A3", "#E58C40"]
    features = ["hidden", "robot_action", "hidden_robot_action", "oracle_metadata"]
    labels = ["Hidden", "Robot + action", "Hidden + R/A", "Condition + time*"]
    for ax, scope in zip(axes[:2], ("all", "glass")):
        for k, target in enumerate(("base_crash", "detour_success")):
            values = [next(r["auc"] for r in metrics if r["regime"] == "pooled_loso" and r["feature"] == f and
                           r["target"] == target and r["scope"] == scope and r["weighting"] == "source") for f in features]
            ax.bar(np.arange(len(features)) + (k - .5) * .35, values, .35,
                   label=("Base crash", "Detour success")[k], color=colors[k])
        ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
        ax.set(ylim=(0, 1.03), xticks=np.arange(len(labels)), xticklabels=labels,
               title="All decisions (n=273)" if scope == "all" else "Glass only (n=101)", ylabel="Source-weighted OOF AUROC")
        ax.tick_params(axis="x", rotation=32)
    axes[0].legend(fontsize=8, loc="lower left")
    rows = [r for r in metrics if r["regime"] == "pooled_loso" and r["feature"] == "hidden" and r["target"] == "detour_success"
            and r["scope"].startswith("glass_T-") and r["weighting"] == "source"]
    rows.sort(key=lambda r: -int(r["scope"].split("-")[1]))
    axes[2].plot(range(len(rows)), [r["auc"] for r in rows], "o-", color=colors[1])
    axes[2].axhline(.5, color="gray", linestyle="--", linewidth=1)
    axes[2].set(xticks=range(len(rows)), xticklabels=[r["scope"].replace("glass_", "") for r in rows],
                ylim=(0, 1.03), title="Hidden: Detour ranking within glass", ylabel="Source-weighted OOF AUROC")
    fig.suptitle("Fixed Detour completion is a separate prediction target\n20-source LOSO; *privileged diagnostic; single-execution development labels", fontsize=12)
    fig.savefig(output / "probe_summary.png", dpi=170)
    fig.savefig(output / "probe_summary.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("research fitting must run in Quest Slurm")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ.get("CB_CODE_COMMIT"):
        raise RuntimeError("submitted commit mismatch")
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], text=True).strip():
        raise RuntimeError("dirty source checkout")
    config = json.loads(args.config.read_text())
    data, validation = validate_capture(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    dump_json(output / "config.json", config)
    provenance = {"started_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
                  "config_sha256": sha256(args.config), "input_sha256": config["input_sha256"],
                  "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")},
                  "python": platform.python_version(), "numpy": np.__version__, "validation": validation,
                  "evidence_role": config["evidence_role"], "new_rollouts": 0, "D8_loaded": False}
    dump_json(output / "provenance.json", provenance)
    predictions = fit_suite(data, config, output)
    dump_csv(output / "predictions.csv", predictions)
    metrics = summarize(predictions, output)
    inference(predictions, config, output)
    plot_results(metrics, output)
    manifest = {"completed_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
                "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(output.iterdir())}}
    dump_json(output / "complete.json", manifest)
    print("DONE " + str(output), flush=True)


if __name__ == "__main__":
    main()
