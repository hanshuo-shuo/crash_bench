#!/usr/bin/env python3
"""Run the closed, existing-data-only Phase 2.5B-R story resolver.

The resolver is a post-hoc protocol amendment because the original Phase 2.5B
gate was not exhaustive in the observed grey region.  It never launches a
simulator, never reads a test/fresh-test split, and never uses an outer source
for PCA, scaling, fitting, or threshold calibration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.counterfactual_router import OPTIONS
from scripts.iclr27.audit_option_support import strict_option_labels
from scripts.iclr27.train_router_baseline_suite import (
    oracle_choice_base_favoring,
    realized_utility,
)
from scripts.iclr27.train_support_crossfit_suite import (
    _apply_threshold,
    _feature_views,
    _preflight_inputs,
    constrained_calibration,
    fit_ridge_weighted,
    outer_source_split,
    predict_ridge,
)
from scripts.train_minimal_counterfactual_router import (
    _source_weights,
    fit_frame_pca,
    load_capture,
)


ADR_METHODS = (
    "ADR-linear-full",
    "ADR-linear-split",
    "ADR-tiny-nonlinear-choice",
)
LINEAR_METHODS = ("ADR-linear-full", "ADR-linear-split")
CHOICE_DIAGNOSTICS = (
    "Fixed-Detour diagnostic",
    "Condition-only choice diagnostic",
    "Horizon-only choice diagnostic",
)
PRIMARY_DECISIONS = (
    "METHOD_READY",
    "CHOICE_CAPACITY_BOTTLENECK",
    "GATE_OR_CALIBRATION_BOTTLENECK",
    "LABEL_MARGIN_BOTTLENECK",
    "CURRENT_BENCHMARK_SIGNAL_INSUFFICIENT",
)
STRICT_LABEL_TO_OPTION = {
    "strict_base": 0,
    "strict_detour": 1,
    "strict_retreat": 2,
}
DEPLOYABLE_FEATURE_FIELDS = ("hidden", "robot_state", "nominal_action")
ORACLE_FIELDS = (
    "U_B", "U_D", "U_R", "intervention_advantage", "choice_advantage",
    "oracle_gate", "oracle_choice",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("")
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_native(materialized))


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("advantage-router config must be a mapping")
    if config.get("kind") != "iclr27_advantage_router_resolver":
        raise ValueError("unexpected advantage-router config kind")
    if tuple(config["decision"]["primary_decisions"]) != PRIMARY_DECISIONS:
        raise ValueError("the exhaustive primary-decision enumeration changed")
    if float(config["utility"]["lambda"]) != 1.0:
        raise ValueError("Phase 2.5B-R freezes lambda=1")
    if float(config["utility"]["eta"]) != 0.0:
        raise ValueError("Phase 2.5B-R freezes eta=0")
    if float(config["calibration"]["maximum_intervention_rate"]) != 0.60:
        raise ValueError("Phase 2.5B-R freezes rho=0.60")
    if tuple(config["utility"]["option_order"]) != tuple(OPTIONS):
        raise ValueError("the frozen option order changed")
    if set(DEPLOYABLE_FEATURE_FIELDS) & set(ORACLE_FIELDS):
        raise AssertionError("oracle information entered deployable feature fields")
    forbidden = set(config["features"]["forbidden_deployable_metadata"])
    if not set(ORACLE_FIELDS) <= forbidden:
        raise ValueError("all oracle targets must be explicitly forbidden features")
    return config


def compute_advantages(utility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return A_int=max(U_D,U_R)-U_B and A_DR=U_D-U_R."""

    utility = np.asarray(utility, dtype=np.float64)
    if utility.ndim != 2 or utility.shape[1] != 3:
        raise ValueError("advantage targets require a [decision, 3-option] matrix")
    return np.max(utility[:, 1:], axis=1) - utility[:, 0], utility[:, 1] - utility[:, 2]


def route_advantages(
    predicted_a_int: np.ndarray, predicted_a_dr: np.ndarray, tau: float
) -> np.ndarray:
    """Apply the frozen Base/Detour/Retreat routing rule."""

    predicted_a_int = np.asarray(predicted_a_int, dtype=np.float64)
    predicted_a_dr = np.asarray(predicted_a_dr, dtype=np.float64)
    if predicted_a_int.shape != predicted_a_dr.shape:
        raise ValueError("advantage heads must produce aligned vectors")
    intervention = np.where(predicted_a_dr >= 0.0, 1, 2)
    return np.where(predicted_a_int > float(tau), intervention, 0).astype(np.int64)


def validate_outer_partition(
    sources: np.ndarray,
    held_source: str,
    fit_sources: Sequence[str],
    calibration_sources: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint masks and fail if the outer source leaks inward."""

    sources = np.asarray(sources, dtype=str)
    fit_names = set(map(str, fit_sources))
    calibration_names = set(map(str, calibration_sources))
    if held_source in fit_names or held_source in calibration_names:
        raise ValueError("outer source entered fitting or calibration")
    if fit_names & calibration_names:
        raise ValueError("fitting and calibration sources overlap")
    held = sources == str(held_source)
    fit = np.isin(sources, sorted(fit_names))
    calibration = np.isin(sources, sorted(calibration_names))
    if not np.any(held):
        raise ValueError("outer source has no decisions")
    if np.any(held & (fit | calibration)) or np.any(fit & calibration):
        raise AssertionError("outer masks overlap")
    if not np.all(held | fit | calibration):
        raise ValueError("outer partition does not cover the corpus")
    return held, fit, calibration


def ensure_fresh_output(output_dir: Path, protected_roots: Sequence[Path]) -> None:
    """Protect frozen Phase 2.5A/2.5B roots and require a new result directory."""

    resolved = output_dir.resolve()
    for root in protected_roots:
        protected = root.resolve()
        if resolved == protected or protected in resolved.parents or resolved in protected.parents:
            raise ValueError(f"refusing to overwrite frozen result root: {protected}")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {resolved}")


def _standardize_weighted(
    x: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance) + 1e-6
    return (x - mean) / scale, mean, scale


def fit_tiny_mlp(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    hidden_units: int,
    l2: float,
    maximum_iterations: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the fixed deterministic one-hidden-layer tanh capacity diagnostic."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or weights.shape != (len(x),):
        raise ValueError("tiny MLP inputs are not aligned")
    xs, mean, scale = _standardize_weighted(x, weights)
    d = xs.shape[1]
    h = int(hidden_units)
    rng = np.random.default_rng(int(seed))
    w1 = rng.normal(0.0, 1.0 / np.sqrt(max(d, 1)), size=(d, h))
    b1 = np.zeros(h, dtype=np.float64)
    w2 = rng.normal(0.0, 1.0 / np.sqrt(max(h, 1)), size=h)
    b2 = np.asarray([np.average(y, weights=weights)], dtype=np.float64)
    initial = np.concatenate((w1.ravel(), b1, w2, b2))
    normalizer = float(weights.sum())

    def unpack(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        offset = d * h
        left = flat[:offset].reshape(d, h)
        hidden_bias = flat[offset:offset + h]
        output_weight = flat[offset + h:offset + 2 * h]
        output_bias = float(flat[-1])
        return left, hidden_bias, output_weight, output_bias

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        left, hidden_bias, output_weight, output_bias = unpack(flat)
        hidden = np.tanh(xs @ left + hidden_bias)
        predicted = hidden @ output_weight + output_bias
        residual = predicted - y
        loss = 0.5 * float(np.sum(weights * residual ** 2)) / normalizer
        loss += 0.5 * float(l2) * (
            float(np.sum(left ** 2)) + float(np.sum(output_weight ** 2))
        )
        output_gradient = weights * residual / normalizer
        grad_w2 = hidden.T @ output_gradient + float(l2) * output_weight
        grad_b2 = float(output_gradient.sum())
        hidden_gradient = (
            output_gradient[:, None] * output_weight[None, :] * (1.0 - hidden ** 2)
        )
        grad_w1 = xs.T @ hidden_gradient + float(l2) * left
        grad_b1 = hidden_gradient.sum(axis=0)
        gradient = np.concatenate((grad_w1.ravel(), grad_b1, grad_w2, [grad_b2]))
        return loss, gradient

    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maximum_iterations), "ftol": 1e-12, "gtol": 1e-8},
    )
    reached_iteration_limit = (
        not fitted.success
        and int(fitted.status) == 1
        and "ITERATIONS REACHED LIMIT" in str(fitted.message)
    )
    if (
        (not fitted.success and not reached_iteration_limit)
        or not np.isfinite(float(fitted.fun))
        or not np.all(np.isfinite(fitted.x))
        or not np.all(np.isfinite(fitted.jac))
    ):
        raise RuntimeError(f"tiny MLP fit failed: {fitted.message}")
    left, hidden_bias, output_weight, output_bias = unpack(fitted.x)
    model = {
        "mean": mean,
        "scale": scale,
        "w1": left,
        "b1": hidden_bias,
        "w2": output_weight,
        "b2": np.asarray(output_bias),
    }
    summary = {
        "converged": bool(fitted.success),
        "stopped_at_iteration_limit": bool(reached_iteration_limit),
        "iterations": int(fitted.nit),
        "objective": float(fitted.fun),
        "seed": int(seed),
        "hidden_units": h,
        "activation": "tanh",
    }
    return model, summary


def predict_tiny_mlp(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    hidden = np.tanh(xs @ model["w1"] + model["b1"])
    return hidden @ model["w2"] + float(np.asarray(model["b2"]))


def _categorical_choice_score(
    categories: np.ndarray,
    fit: np.ndarray,
    target: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    categories = np.asarray(categories, dtype=str)
    target = np.asarray(target, dtype=np.float64)
    fit_weights = _source_weights(sources[fit])
    global_mean = float(np.average(target[fit], weights=fit_weights))
    result = np.full(len(target), global_mean, dtype=np.float64)
    for category in sorted(set(categories.tolist())):
        local = fit & (categories == category)
        if np.any(local):
            result[categories == category] = float(np.average(
                target[local], weights=_source_weights(sources[local])
            ))
    return result


def _verify_frozen_crossfit(
    path: Path,
    config: Mapping[str, Any],
    capture_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    for name in config["inputs"]["required_crossfit_files"]:
        if not (path / str(name)).is_file():
            raise FileNotFoundError(path / str(name))
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("kind") != "iclr27_support_crossfit_suite":
        raise ValueError("reused result is not the frozen Phase 2.5B suite")
    gate = json.loads((path / "gate_decision.json").read_text())
    required_gate = str(config["inputs"]["required_crossfit_gate"])
    if gate.get("decision") != required_gate or manifest.get("gate_decision") != required_gate:
        raise ValueError("Phase 2.5B frozen gate changed")
    hashes: dict[str, str] = {}
    for name in config["inputs"]["required_crossfit_files"]:
        actual = _sha256(path / str(name))
        hashes[f"phase_2_5b/{name}"] = actual
        if name != "manifest.json" and manifest["artifact_sha256"].get(name) != actual:
            raise ValueError(f"frozen Phase 2.5B artifact hash mismatch: {name}")
    for key, actual in capture_hashes.items():
        if key.startswith("capture/") and manifest["input_sha256"].get(key) != actual:
            raise ValueError(f"capture fingerprint differs from Phase 2.5B: {key}")
    archive_path = path / "all_predictions.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required_methods = tuple(map(str, config["reused_phase_2_5b_methods"]))
    observed_methods = tuple(map(str, arrays["method_names"].tolist()))
    if observed_methods != required_methods:
        raise ValueError("the frozen Phase 2.5B method order changed")
    return manifest, arrays, hashes


def _source_equal_weights(
    sources: np.ndarray,
    mask: np.ndarray | None = None,
    multiplicity: np.ndarray | None = None,
    source_levels: Sequence[str] | None = None,
) -> np.ndarray:
    sources = np.asarray(sources, dtype=str)
    selected = np.ones(len(sources), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    local_sources = sources[selected]
    if not len(local_sources):
        return np.asarray([], dtype=np.float64)
    levels = list(source_levels) if source_levels is not None else sorted(set(sources.tolist()))
    level_index = {str(source): index for index, source in enumerate(levels)}
    if multiplicity is None:
        multiplicity = np.ones(len(levels), dtype=np.float64)
    multiplicity = np.asarray(multiplicity, dtype=np.float64)
    counts = Counter(local_sources.tolist())
    weights = np.asarray([
        multiplicity[level_index[source]] / counts[source] for source in local_sources
    ], dtype=np.float64)
    return weights


def _weighted_mean(
    values: np.ndarray,
    sources: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    multiplicity: np.ndarray | None = None,
    source_levels: Sequence[str] | None = None,
) -> float:
    values = np.asarray(values)
    selected = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return float("nan")
    weights = _source_equal_weights(
        sources, selected, multiplicity=multiplicity, source_levels=source_levels
    )
    if float(weights.sum()) <= 0.0:
        return float("nan")
    return float(np.average(values[selected].astype(np.float64), weights=weights))


def _weighted_binary_auc(
    target: np.ndarray,
    score: np.ndarray,
    sources: np.ndarray,
    *,
    multiplicity: np.ndarray | None = None,
    source_levels: Sequence[str] | None = None,
) -> tuple[float, float]:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    weights = _source_equal_weights(
        sources, multiplicity=multiplicity, source_levels=source_levels
    )
    positive = target
    negative = ~target
    positive_mass = float(weights[positive].sum())
    negative_mass = float(weights[negative].sum())
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        auroc = float("nan")
    else:
        comparison = score[positive, None] - score[negative][None, :]
        pair_weights = weights[positive, None] * weights[negative][None, :]
        credit = (comparison > 0.0).astype(np.float64) + 0.5 * (comparison == 0.0)
        auroc = float(np.sum(pair_weights * credit) / (positive_mass * negative_mass))
    if positive_mass <= 0.0:
        auprc = float("nan")
    else:
        order = np.argsort(-score, kind="mergesort")
        sorted_score = score[order]
        sorted_positive = target[order]
        sorted_weights = weights[order]
        group_end = np.r_[
            np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(order) - 1
        ]
        tp = np.cumsum(sorted_weights * sorted_positive)[group_end]
        fp = np.cumsum(sorted_weights * ~sorted_positive)[group_end]
        recall = tp / positive_mass
        precision = tp / np.maximum(tp + fp, 1e-15)
        recall_previous = np.concatenate(([0.0], recall[:-1]))
        auprc = float(np.sum((recall - recall_previous) * precision))
    return auroc, auprc


def conditional_choice_metrics(
    predicted_choice: np.ndarray,
    target_choice: np.ndarray,
    sources: np.ndarray,
    mask: np.ndarray,
    *,
    multiplicity: np.ndarray | None = None,
    source_levels: Sequence[str] | None = None,
) -> dict[str, Any]:
    predicted_choice = np.asarray(predicted_choice, dtype=np.int64)
    target_choice = np.asarray(target_choice, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    weights = _source_equal_weights(
        sources, mask, multiplicity=multiplicity, source_levels=source_levels
    )
    target = target_choice[mask]
    predicted = predicted_choice[mask]
    if not len(target) or float(weights.sum()) <= 0.0:
        return {
            "balanced_accuracy": float("nan"), "macro_f1": float("nan"),
            "detour_recall": float("nan"), "retreat_recall": float("nan"),
            "raw_decisions": int(len(target)),
        }
    confusion_weighted = np.zeros((2, 2), dtype=np.float64)
    confusion_raw = np.zeros((2, 2), dtype=np.int64)
    for target_option, target_index in ((1, 0), (2, 1)):
        for predicted_option, predicted_index in ((1, 0), (2, 1)):
            local = (target == target_option) & (predicted == predicted_option)
            confusion_weighted[target_index, predicted_index] = float(weights[local].sum())
            confusion_raw[target_index, predicted_index] = int(np.sum(local))
    recalls = []
    f1s = []
    for index in range(2):
        tp = confusion_weighted[index, index]
        fn = float(confusion_weighted[index, :].sum() - tp)
        fp = float(confusion_weighted[:, index].sum() - tp)
        recalls.append(tp / (tp + fn) if tp + fn > 0.0 else float("nan"))
        f1s.append(2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn > 0.0 else 0.0)
    return {
        "balanced_accuracy": float(np.nanmean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "detour_recall": float(recalls[0]),
        "retreat_recall": float(recalls[1]),
        "raw_decisions": int(len(target)),
        "raw_target_detour_pred_detour": int(confusion_raw[0, 0]),
        "raw_target_detour_pred_retreat": int(confusion_raw[0, 1]),
        "raw_target_retreat_pred_detour": int(confusion_raw[1, 0]),
        "raw_target_retreat_pred_retreat": int(confusion_raw[1, 1]),
        "weighted_confusion_matrix": json.dumps(_native(confusion_weighted.tolist())),
    }


def gate_metrics(
    score: np.ndarray,
    choice: np.ndarray,
    a_int: np.ndarray,
    sources: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    multiplicity: np.ndarray | None = None,
    source_levels: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = np.ones(len(sources), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    target = np.asarray(a_int, dtype=np.float64) > 0.0
    predicted = np.asarray(choice, dtype=np.int64) != 0
    base_mask = selected & ~target
    intervention_mask = selected & target
    base_recall = _weighted_mean(
        ~predicted, sources, base_mask, multiplicity=multiplicity, source_levels=source_levels
    )
    intervention_recall = _weighted_mean(
        predicted, sources, intervention_mask,
        multiplicity=multiplicity, source_levels=source_levels,
    )
    auroc, auprc = _weighted_binary_auc(
        target[selected], np.asarray(score)[selected], np.asarray(sources)[selected],
        multiplicity=multiplicity, source_levels=source_levels,
    )
    return {
        "base_recall": base_recall,
        "intervention_recall": intervention_recall,
        "balanced_accuracy": float(np.nanmean([base_recall, intervention_recall])),
        "auroc": auroc,
        "auprc": auprc,
        "intervention_rate": _weighted_mean(
            predicted, sources, selected,
            multiplicity=multiplicity, source_levels=source_levels,
        ),
        "raw_decisions": int(selected.sum()),
        "raw_base_states": int(base_mask.sum()),
        "raw_intervention_states": int(intervention_mask.sum()),
    }


def composed_metrics(
    choice: np.ndarray,
    outcomes: np.ndarray,
    utility: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    multiplicity: np.ndarray | None = None,
    source_levels: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = np.ones(len(sources), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    row_indices = np.arange(len(choice))
    selected_outcome = outcomes[row_indices, choice]
    selected_utility = utility[row_indices, choice]
    result: dict[str, Any] = {
        "scalar_utility": _weighted_mean(
            selected_utility, sources, selected,
            multiplicity=multiplicity, source_levels=source_levels,
        ),
        "task_success_rate": _weighted_mean(
            selected_outcome == "task_success", sources, selected,
            multiplicity=multiplicity, source_levels=source_levels,
        ),
        "catastrophe_rate": _weighted_mean(
            selected_outcome == "catastrophe", sources, selected,
            multiplicity=multiplicity, source_levels=source_levels,
        ),
        "intervention_rate": _weighted_mean(
            choice != 0, sources, selected,
            multiplicity=multiplicity, source_levels=source_levels,
        ),
        "raw_decisions": int(selected.sum()),
    }
    recalls: dict[str, float] = {}
    for label, option in STRICT_LABEL_TO_OPTION.items():
        local = selected & (labels == label)
        recalls[label] = _weighted_mean(
            choice == option, sources, local,
            multiplicity=multiplicity, source_levels=source_levels,
        )
        result[f"{label}_recall"] = recalls[label]
        result[f"raw_{label}_states"] = int(local.sum())
    result["detour_retreat_balanced_accuracy"] = float(np.nanmean([
        recalls["strict_detour"], recalls["strict_retreat"]
    ]))
    strict = selected & np.isin(labels, list(STRICT_LABEL_TO_OPTION))
    strict_weights = _source_equal_weights(
        sources, strict, multiplicity=multiplicity, source_levels=source_levels
    )
    strict_target = np.asarray([
        STRICT_LABEL_TO_OPTION[str(label)] for label in labels[strict]
    ], dtype=np.int64)
    strict_predicted = choice[strict]
    f1s = []
    for option in range(3):
        tp = float(strict_weights[(strict_target == option) & (strict_predicted == option)].sum())
        fp = float(strict_weights[(strict_target != option) & (strict_predicted == option)].sum())
        fn = float(strict_weights[(strict_target == option) & (strict_predicted != option)].sum())
        denominator = 2.0 * tp + fp + fn
        f1s.append(0.0 if denominator <= 0.0 else 2.0 * tp / denominator)
    result["strict_choice_macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    return result


def choice_pass_check(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    criteria = {
        "conditional_dr_balanced_accuracy": float(metrics["balanced_accuracy"]) >= float(
            thresholds["conditional_dr_balanced_accuracy_min"]
        ),
        "strict_detour_recall": float(metrics["detour_recall"]) >= float(
            thresholds["strict_detour_recall_min"]
        ),
        "strict_retreat_recall": float(metrics["retreat_recall"]) >= float(
            thresholds["strict_retreat_recall_min"]
        ),
    }
    return {"passes": bool(all(criteria.values())), "criteria": criteria, "actual": dict(metrics)}


def method_pass_check(
    *,
    method: str,
    metrics: Mapping[str, Any],
    risk_metrics: Mapping[str, Any],
    source_utility: Mapping[tuple[str, str], float],
    sources: Sequence[str],
    comparator_metrics: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    utility_gain = float(metrics["scalar_utility"]) - float(risk_metrics["scalar_utility"])
    rate_gain = float(metrics["intervention_rate"]) - float(risk_metrics["intervention_rate"])
    catastrophe_gain = float(metrics["catastrophe_rate"]) - float(risk_metrics["catastrophe_rate"])
    nonnegative = sum(
        source_utility[(method, source)] - source_utility[(str(thresholds["risk_reference"]), source)] >= -1e-15
        for source in sources
    )
    comparator_checks = []
    for comparator in thresholds["support_balanced_comparators"]:
        right = comparator_metrics[str(comparator)]
        comparator_utility_gain = float(metrics["scalar_utility"]) - float(right["scalar_utility"])
        f1_gain = float(metrics["strict_choice_macro_f1"]) - float(right["strict_choice_macro_f1"])
        passes = (
            comparator_utility_gain >= float(thresholds["support_comparator_utility_gain_min"])
            or (
                f1_gain >= float(thresholds["support_comparator_macro_f1_gain_min"])
                and comparator_utility_gain >= -float(thresholds["support_comparator_utility_drop_max"])
            )
        )
        comparator_checks.append({
            "comparator": str(comparator),
            "utility_gain": comparator_utility_gain,
            "strict_choice_macro_f1_gain": f1_gain,
            "passes": bool(passes),
        })
    criteria = {
        "strict_detour_recall": float(metrics["strict_detour_recall"]) >= float(thresholds["strict_detour_recall_min"]),
        "strict_retreat_recall": float(metrics["strict_retreat_recall"]) >= float(thresholds["strict_retreat_recall_min"]),
        "strict_base_recall": float(metrics["strict_base_recall"]) >= float(thresholds["strict_base_recall_min"]),
        "utility_gain_over_risk": utility_gain >= float(thresholds["utility_gain_over_risk_min"]),
        "intervention_rate_cap": float(metrics["intervention_rate"]) <= float(thresholds["intervention_rate_cap"]) + 1e-12,
        "intervention_rate_vs_risk": rate_gain <= float(thresholds["intervention_rate_increase_over_risk_max"]),
        "catastrophe_rate_vs_risk": catastrophe_gain <= float(thresholds["catastrophe_rate_increase_over_risk_max"]),
        "nonnegative_paired_sources": nonnegative >= int(thresholds["nonnegative_paired_sources_min"]),
        "support_balanced_comparator": any(row["passes"] for row in comparator_checks),
    }
    return {
        "passes": bool(all(criteria.values())),
        "criteria": criteria,
        "actual": {
            "strict_detour_recall": metrics["strict_detour_recall"],
            "strict_retreat_recall": metrics["strict_retreat_recall"],
            "strict_base_recall": metrics["strict_base_recall"],
            "utility_gain_over_risk": utility_gain,
            "intervention_rate": metrics["intervention_rate"],
            "intervention_rate_increase_over_risk": rate_gain,
            "catastrophe_rate_increase_over_risk": catastrophe_gain,
            "nonnegative_paired_sources": int(nonnegative),
        },
        "support_comparator_checks": comparator_checks,
    }


def select_primary_decision(
    *,
    method_pass: Mapping[str, bool],
    choice_pass: Mapping[str, bool],
    upper_half_choice_pass: Mapping[str, bool],
) -> str:
    """Apply the predeclared ordered resolver; INCONCLUSIVE is impossible."""

    if any(bool(method_pass.get(method, False)) for method in ADR_METHODS):
        return "METHOD_READY"
    if (
        bool(choice_pass.get("ADR-tiny-nonlinear-choice", False))
        and not any(bool(choice_pass.get(method, False)) for method in LINEAR_METHODS)
    ):
        return "CHOICE_CAPACITY_BOTTLENECK"
    if any(bool(choice_pass.get(method, False)) for method in ADR_METHODS):
        return "GATE_OR_CALIBRATION_BOTTLENECK"
    if (
        not any(bool(choice_pass.get(method, False)) for method in ADR_METHODS)
        and any(bool(upper_half_choice_pass.get(method, False)) for method in ADR_METHODS)
    ):
        return "LABEL_MARGIN_BOTTLENECK"
    return "CURRENT_BENCHMARK_SIGNAL_INSUFFICIENT"


def _interval(values: Sequence[float]) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None, None
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def _bootstrap_metric_rows(
    *,
    methods: Sequence[str],
    choices: Mapping[str, np.ndarray],
    predicted_a_int: Mapping[str, np.ndarray],
    predicted_a_dr: Mapping[str, np.ndarray],
    risk_choice: np.ndarray,
    outcomes: np.ndarray,
    utility: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    a_int: np.ndarray,
    target_dr: np.ndarray,
    strict_dr: np.ndarray,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bootstrap = config["statistics"]["bootstrap"]
    levels = sorted(set(sources.tolist()))
    rng = np.random.default_rng(int(bootstrap["seed"]))
    draws = rng.integers(0, len(levels), size=(int(bootstrap["replicates"]), len(levels)))
    counts = np.zeros((len(draws), len(levels)), dtype=np.int16)
    for index in range(len(levels)):
        counts[:, index] = np.sum(draws == index, axis=1)

    rows: list[dict[str, Any]] = []
    risk_boot = []
    for multiplicity in counts:
        risk_boot.append(composed_metrics(
            risk_choice, outcomes, utility, labels, sources,
            multiplicity=multiplicity, source_levels=levels,
        )["scalar_utility"])
    risk_boot_array = np.asarray(risk_boot, dtype=np.float64)

    composed_names = list(choices)
    for method in composed_names:
        estimates = composed_metrics(choices[method], outcomes, utility, labels, sources)
        distributions: dict[str, list[float]] = {
            metric: [] for metric in (
                "scalar_utility", "catastrophe_rate", "intervention_rate",
                "strict_base_recall", "strict_detour_recall", "strict_retreat_recall",
            )
        }
        gain_distribution = []
        for draw_index, multiplicity in enumerate(counts):
            current = composed_metrics(
                choices[method], outcomes, utility, labels, sources,
                multiplicity=multiplicity, source_levels=levels,
            )
            for metric in distributions:
                distributions[metric].append(float(current[metric]))
            gain_distribution.append(float(current["scalar_utility"]) - risk_boot_array[draw_index])
        for metric, values in distributions.items():
            lower, upper = _interval(values)
            rows.append({
                "component": "composed_router", "method": method, "metric": metric,
                "estimate": estimates[metric], "ci_lower": lower, "ci_upper": upper,
                "bootstrap_unit": "source", "replicates": len(draws),
            })
        lower, upper = _interval(gain_distribution)
        risk_estimate = composed_metrics(risk_choice, outcomes, utility, labels, sources)["scalar_utility"]
        rows.append({
            "component": "composed_router", "method": method,
            "metric": "utility_gain_vs_Risk->BestFixed",
            "estimate": float(estimates["scalar_utility"]) - float(risk_estimate),
            "ci_lower": lower, "ci_upper": upper,
            "bootstrap_unit": "source", "replicates": len(draws),
        })

    for method in methods:
        choice_estimate = conditional_choice_metrics(
            np.where(predicted_a_dr[method] >= 0.0, 1, 2), target_dr,
            sources, strict_dr,
        )
        gate_estimate = gate_metrics(
            predicted_a_int[method], choices[method], a_int, sources
        )
        choice_distributions = {metric: [] for metric in (
            "balanced_accuracy", "macro_f1", "detour_recall", "retreat_recall"
        )}
        gate_distributions = {metric: [] for metric in (
            "base_recall", "intervention_recall", "balanced_accuracy", "auroc",
            "auprc", "intervention_rate",
        )}
        predicted_choice = np.where(predicted_a_dr[method] >= 0.0, 1, 2)
        for multiplicity in counts:
            choice_current = conditional_choice_metrics(
                predicted_choice, target_dr, sources, strict_dr,
                multiplicity=multiplicity, source_levels=levels,
            )
            gate_current = gate_metrics(
                predicted_a_int[method], choices[method], a_int, sources,
                multiplicity=multiplicity, source_levels=levels,
            )
            for metric in choice_distributions:
                choice_distributions[metric].append(float(choice_current[metric]))
            for metric in gate_distributions:
                gate_distributions[metric].append(float(gate_current[metric]))
        for metric, values in choice_distributions.items():
            lower, upper = _interval(values)
            rows.append({
                "component": "conditional_choice", "method": method, "metric": metric,
                "estimate": choice_estimate[metric], "ci_lower": lower, "ci_upper": upper,
                "bootstrap_unit": "source", "replicates": len(draws),
            })
        for metric, values in gate_distributions.items():
            lower, upper = _interval(values)
            rows.append({
                "component": "benefit_gate", "method": method, "metric": metric,
                "estimate": gate_estimate[metric], "ci_lower": lower, "ci_upper": upper,
                "bootstrap_unit": "source", "replicates": len(draws),
            })
    return rows


def _group_metric_rows(
    *,
    group_type: str,
    group_values: np.ndarray,
    choices: Mapping[str, np.ndarray],
    choice_predictions: Mapping[str, np.ndarray],
    predicted_a_int: Mapping[str, np.ndarray],
    outcomes: np.ndarray,
    utility: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    a_int: np.ndarray,
    target_dr: np.ndarray,
    strict_dr: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_values = np.asarray(group_values, dtype=str)
    for value in sorted(set(group_values.tolist())):
        mask = group_values == value
        for method, choice in choices.items():
            rows.append({
                "component": "composed_router", "group_type": group_type,
                "group_value": value, "method": method,
                **composed_metrics(choice, outcomes, utility, labels, sources, mask=mask),
            })
        for method, predicted_choice in choice_predictions.items():
            local = mask & strict_dr
            rows.append({
                "component": "conditional_choice", "group_type": group_type,
                "group_value": value, "method": method,
                **conditional_choice_metrics(predicted_choice, target_dr, sources, local),
            })
        for method, score in predicted_a_int.items():
            rows.append({
                "component": "benefit_gate", "group_type": group_type,
                "group_value": value, "method": method,
                **gate_metrics(score, choices[method], a_int, sources, mask=mask),
            })
    return rows


def _source_metric_rows(
    *,
    choices: Mapping[str, np.ndarray],
    choice_predictions: Mapping[str, np.ndarray],
    predicted_a_int: Mapping[str, np.ndarray],
    outcomes: np.ndarray,
    utility: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    a_int: np.ndarray,
    target_dr: np.ndarray,
    strict_dr: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        for method, choice in choices.items():
            rows.append({
                "component": "composed_router", "source": source, "method": method,
                **composed_metrics(choice, outcomes, utility, labels, sources, mask=mask),
            })
        for method, predicted_choice in choice_predictions.items():
            rows.append({
                "component": "conditional_choice", "source": source, "method": method,
                **conditional_choice_metrics(
                    predicted_choice, target_dr, sources, mask & strict_dr
                ),
            })
        for method, score in predicted_a_int.items():
            rows.append({
                "component": "benefit_gate", "source": source, "method": method,
                **gate_metrics(score, choices[method], a_int, sources, mask=mask),
            })
    return rows


def run_resolver(
    *,
    config_path: Path,
    capture: Path,
    option_support: Path,
    support_crossfit: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)
    capture = capture.resolve()
    option_support = option_support.resolve()
    support_crossfit = support_crossfit.resolve()
    output_dir = output_dir.resolve()
    ensure_fresh_output(output_dir, (option_support, support_crossfit))

    metadata_preflight, input_hashes = _preflight_inputs(capture, option_support, config)
    frozen_manifest, frozen, crossfit_hashes = _verify_frozen_crossfit(
        support_crossfit, config, input_hashes
    )
    input_hashes.update(crossfit_hashes)
    output_dir.mkdir(parents=True, exist_ok=False)

    data = load_capture(capture, float(config["utility"]["lambda"]))
    if len(metadata_preflight) != len(data["metadata"]):
        raise ValueError("capture metadata changed after preflight")
    expected_decisions = int(config["data_policy"]["expected_decision_count"])
    if len(data["metadata"]) != expected_decisions:
        raise ValueError(f"expected {expected_decisions} decisions")
    sources = np.asarray(data["sources"], dtype=str)
    outcomes = np.asarray(data["outcomes"], dtype=str)
    utility = realized_utility(
        outcomes, float(config["utility"]["lambda"]), float(config["utility"]["eta"])
    )
    labels, strict_advantage = strict_option_labels(
        utility, catastrophe_utility=-float(config["utility"]["lambda"])
    )
    a_int, a_dr = compute_advantages(utility)
    target_dr = np.where(a_dr >= 0.0, 1, 2).astype(np.int64)
    strict_dr = np.isin(labels, ["strict_detour", "strict_retreat"])
    oracle_gap_selected = (a_int > 0.0) & (np.abs(a_dr) > 1e-12)
    oracle_choice = np.where(a_int > 0.0, target_dr, 0).astype(np.int64)
    decision_ids = np.asarray([str(row["decision_id"]) for row in data["metadata"]])
    families = np.asarray(data["conditions"], dtype=str)
    horizons = np.asarray(data["horizons"], dtype=np.int64)
    historical_split = np.asarray(data["splits"], dtype=str)
    unique_sources = sorted(set(sources.tolist()))
    if len(unique_sources) != int(config["data_policy"]["expected_source_count"]):
        raise ValueError("source count changed")

    if not np.array_equal(np.asarray(frozen["decision_id"], dtype=str), decision_ids):
        raise ValueError("frozen Phase 2.5B decision order differs from the capture")
    if not np.array_equal(np.asarray(frozen["source"], dtype=str), sources):
        raise ValueError("frozen Phase 2.5B source order differs from the capture")
    if not np.array_equal(np.asarray(frozen["outcomes"], dtype=str), outcomes):
        raise ValueError("frozen Phase 2.5B outcomes differ from the capture")

    n = len(sources)
    predicted_a_int = {method: np.full(n, np.nan) for method in ADR_METHODS}
    predicted_a_dr = {method: np.full(n, np.nan) for method in ADR_METHODS}
    deploy_choice = {method: np.full(n, -1, dtype=np.int64) for method in ADR_METHODS}
    hybrid_choice: dict[tuple[str, str], np.ndarray] = {}
    for method in ADR_METHODS:
        for hybrid in (
            "OracleGate+OracleChoice", "OracleGate+LearnedChoice",
            "LearnedGate+OracleChoice", "LearnedGate+LearnedChoice",
        ):
            hybrid_choice[(method, hybrid)] = np.full(n, -1, dtype=np.int64)
    diagnostic_choice = {
        "Fixed-Detour diagnostic": np.ones(n, dtype=np.int64),
        "Condition-only choice diagnostic": np.full(n, -1, dtype=np.int64),
        "Horizon-only choice diagnostic": np.full(n, -1, dtype=np.int64),
    }
    fold_rows: list[dict[str, Any]] = []
    calibration_records: dict[str, Any] = {}
    fit_summaries: dict[str, Any] = {}
    fold_thresholds: dict[str, dict[str, float]] = {method: {} for method in ADR_METHODS}
    outer = config["data_policy"]["outer_crossfit"]
    seed = int(config["models"]["seed"])
    ridge_alpha = float(config["models"]["ridge_alpha"])
    pca_components = int(config["features"]["pca_components"])
    rho = float(config["calibration"]["maximum_intervention_rate"])
    mlp_config = config["models"]["tiny_mlp"]

    for fold_index, held_source in enumerate(unique_sources):
        fit_names, calibration_names = outer_source_split(
            unique_sources, held_source,
            fit_sources=int(outer["fit_sources"]), seed=int(outer["seed"]),
        )
        held, fit, calibration = validate_outer_partition(
            sources, held_source, fit_names, calibration_names
        )
        for source in unique_sources:
            role = "outer_test" if source == held_source else (
                "fit" if source in fit_names else "calibration"
            )
            fold_rows.append({
                "outer_fold": fold_index, "held_source": held_source,
                "source": source, "role": role,
            })
        pca = fit_frame_pca(
            data["arrays"]["hidden"], data["arrays"]["history_mask"], fit,
            pca_components, seed=seed + fold_index,
        )
        feature_views = _feature_views(data["arrays"], pca)
        x_full = feature_views["full"]
        x_hidden = feature_views["hidden"]
        weights = _source_weights(sources[fit])
        gate_model = fit_ridge_weighted(
            x_full[fit], a_int[fit], weights, alpha=ridge_alpha
        )
        gate_prediction = predict_ridge(gate_model, x_full)
        full_choice_model = fit_ridge_weighted(
            x_full[fit], a_dr[fit], weights, alpha=ridge_alpha
        )
        full_choice_prediction = predict_ridge(full_choice_model, x_full)
        hidden_choice_model = fit_ridge_weighted(
            x_hidden[fit], a_dr[fit], weights, alpha=ridge_alpha
        )
        hidden_choice_prediction = predict_ridge(hidden_choice_model, x_hidden)
        mlp_model, mlp_summary = fit_tiny_mlp(
            x_hidden[fit], a_dr[fit], weights,
            hidden_units=int(mlp_config["hidden_units"]),
            l2=float(mlp_config["l2"]),
            maximum_iterations=int(mlp_config["maximum_iterations"]),
            seed=seed + fold_index,
        )
        mlp_choice_prediction = predict_tiny_mlp(mlp_model, x_hidden)
        choice_predictions = {
            "ADR-linear-full": full_choice_prediction,
            "ADR-linear-split": hidden_choice_prediction,
            "ADR-tiny-nonlinear-choice": mlp_choice_prediction,
        }

        for method, choice_prediction in choice_predictions.items():
            proposed_intervention = np.where(choice_prediction >= 0.0, 1, 2)
            calibration_record = constrained_calibration(
                gate_prediction[calibration], proposed_intervention[calibration],
                utility[calibration], outcomes[calibration], sources[calibration],
                maximum_rate=rho,
            )
            tau = float(calibration_record["threshold"])
            learned = route_advantages(gate_prediction, choice_prediction, tau)
            predicted_a_int[method][held] = gate_prediction[held]
            predicted_a_dr[method][held] = choice_prediction[held]
            deploy_choice[method][held] = learned[held]
            fold_thresholds[method][held_source] = tau
            calibration_records[f"fold_{fold_index:02d}/{method}"] = {
                **calibration_record,
                "outer_test_source": held_source,
                "fit_sources": list(fit_names),
                "calibration_sources": list(calibration_names),
                "outer_test_used": False,
            }
            hybrid_choice[(method, "OracleGate+OracleChoice")][held] = oracle_choice[held]
            hybrid_choice[(method, "OracleGate+LearnedChoice")][held] = np.where(
                a_int[held] > 0.0, proposed_intervention[held], 0
            )
            hybrid_choice[(method, "LearnedGate+OracleChoice")][held] = np.where(
                gate_prediction[held] > tau, target_dr[held], 0
            )
            hybrid_choice[(method, "LearnedGate+LearnedChoice")][held] = learned[held]

        condition_score = _categorical_choice_score(
            families, fit, a_dr, sources
        )
        horizon_score = _categorical_choice_score(
            horizons.astype(str), fit, a_dr, sources
        )
        diagnostic_choice["Condition-only choice diagnostic"][held] = np.where(
            condition_score[held] >= 0.0, 1, 2
        )
        diagnostic_choice["Horizon-only choice diagnostic"][held] = np.where(
            horizon_score[held] >= 0.0, 1, 2
        )
        fit_summaries[f"fold_{fold_index:02d}"] = {
            "held_source": held_source,
            "fit_sources": list(fit_names),
            "calibration_sources": list(calibration_names),
            "pca_fit_decisions": int(fit.sum()),
            "pca_fit_sources": list(fit_names),
            "scaler_fit_sources": list(fit_names),
            "mlp_preprocessing_fit_sources": list(fit_names),
            "mlp": mlp_summary,
        }

    for method in ADR_METHODS:
        if np.any(~np.isfinite(predicted_a_int[method])) or np.any(~np.isfinite(predicted_a_dr[method])):
            raise RuntimeError(f"incomplete OOF scores for {method}")
        if np.any(deploy_choice[method] < 0):
            raise RuntimeError(f"incomplete OOF choice for {method}")
        for hybrid in (
            "OracleGate+OracleChoice", "OracleGate+LearnedChoice",
            "LearnedGate+OracleChoice", "LearnedGate+LearnedChoice",
        ):
            if np.any(hybrid_choice[(method, hybrid)] < 0):
                raise RuntimeError(f"incomplete hybrid choices for {method}/{hybrid}")
    if any(np.any(choice < 0) for choice in diagnostic_choice.values()):
        raise RuntimeError("incomplete OOF diagnostic choice")

    frozen_methods = tuple(map(str, frozen["method_names"].tolist()))
    frozen_choices = np.asarray(frozen["choices"], dtype=np.int64)
    all_composed_choices: dict[str, np.ndarray] = {
        method: frozen_choices[index] for index, method in enumerate(frozen_methods)
    }
    all_composed_choices.update(deploy_choice)
    all_composed_choices["Oracle upper bound diagnostic"] = oracle_choice
    risk_name = str(config["decision"]["method_pass"]["risk_reference"])
    risk_choice = all_composed_choices[risk_name]
    risk_metrics = composed_metrics(risk_choice, outcomes, utility, labels, sources)

    overall_rows: list[dict[str, Any]] = []
    overall_lookup: dict[str, dict[str, Any]] = {}
    for method, choice in all_composed_choices.items():
        metrics = composed_metrics(choice, outcomes, utility, labels, sources)
        metrics["utility_gain_vs_Risk->BestFixed"] = (
            float(metrics["scalar_utility"]) - float(risk_metrics["scalar_utility"])
        )
        role = (
            "advantage_router" if method in ADR_METHODS else
            ("diagnostic_only" if "diagnostic" in method.lower() else "reused_phase_2_5b")
        )
        row = {"method": method, "role": role, "deployable": method in ADR_METHODS or role == "reused_phase_2_5b", **metrics}
        overall_rows.append(row)
        overall_lookup[method] = row

    conditional_rows: list[dict[str, Any]] = []
    choice_prediction_map = {
        **{method: np.where(predicted_a_dr[method] >= 0.0, 1, 2) for method in ADR_METHODS},
        **diagnostic_choice,
    }
    subset_masks = {
        "strict_dr": strict_dr,
        "oracle_gap_selected": oracle_gap_selected,
    }
    choice_metrics_lookup: dict[str, dict[str, Any]] = {}
    comparator_strict: dict[str, dict[str, Any]] = {}
    for method, predicted_choice in choice_prediction_map.items():
        for subset, mask in subset_masks.items():
            metrics = conditional_choice_metrics(
                predicted_choice, target_dr, sources, mask
            )
            row = {"method": method, "subset": subset, **metrics}
            conditional_rows.append(row)
            if subset == "strict_dr":
                if method in ADR_METHODS:
                    choice_metrics_lookup[method] = metrics
                else:
                    comparator_strict[method] = metrics
    for row in conditional_rows:
        if row["method"] not in ADR_METHODS or row["subset"] != "strict_dr":
            continue
        for comparator, comparator_metrics in comparator_strict.items():
            key = comparator.replace(" diagnostic", "").lower().replace("-", "_").replace(" ", "_")
            row[f"balanced_accuracy_delta_vs_{key}"] = (
                float(row["balanced_accuracy"]) - float(comparator_metrics["balanced_accuracy"])
            )
            row[f"macro_f1_delta_vs_{key}"] = (
                float(row["macro_f1"]) - float(comparator_metrics["macro_f1"])
            )

    gate_rows = []
    gate_lookup: dict[str, dict[str, Any]] = {}
    for method in ADR_METHODS:
        metrics = gate_metrics(
            predicted_a_int[method], deploy_choice[method], a_int, sources
        )
        gate_lookup[method] = metrics
        gate_rows.append({"method": method, **metrics})

    source_rows = _source_metric_rows(
        choices=all_composed_choices,
        choice_predictions=choice_prediction_map,
        predicted_a_int=predicted_a_int,
        outcomes=outcomes, utility=utility, labels=labels, sources=sources,
        a_int=a_int, target_dr=target_dr, strict_dr=strict_dr,
    )
    source_utility = {
        (str(row["method"]), str(row["source"])): float(row["scalar_utility"])
        for row in source_rows if row["component"] == "composed_router"
    }
    comparator_metrics = {
        name: overall_lookup[name]
        for name in config["decision"]["method_pass"]["support_balanced_comparators"]
    }
    method_checks: dict[str, dict[str, Any]] = {}
    for method in ADR_METHODS:
        method_checks[method] = method_pass_check(
            method=method, metrics=overall_lookup[method], risk_metrics=risk_metrics,
            source_utility=source_utility, sources=unique_sources,
            comparator_metrics=comparator_metrics,
            thresholds=config["decision"]["method_pass"],
        )

    hybrid_rows: list[dict[str, Any]] = []
    hybrid_method_checks: dict[str, dict[str, Any]] = {}
    hybrid_named_choices: dict[str, np.ndarray] = {}
    for method in ADR_METHODS:
        combo_metrics: dict[str, dict[str, Any]] = {}
        for hybrid in (
            "OracleGate+OracleChoice", "OracleGate+LearnedChoice",
            "LearnedGate+OracleChoice", "LearnedGate+LearnedChoice",
        ):
            hybrid_name = f"{method}/{hybrid}"
            choice = hybrid_choice[(method, hybrid)]
            hybrid_named_choices[hybrid_name] = choice
            metrics = composed_metrics(choice, outcomes, utility, labels, sources)
            metrics["utility_gain_vs_Risk->BestFixed"] = (
                float(metrics["scalar_utility"]) - float(risk_metrics["scalar_utility"])
            )
            combo_metrics[hybrid] = metrics
            if hybrid == "LearnedGate+OracleChoice":
                hybrid_source_utility = {}
                for source in unique_sources:
                    local = sources == source
                    hybrid_source_utility[(hybrid_name, source)] = float(composed_metrics(
                        choice, outcomes, utility, labels, sources, mask=local
                    )["scalar_utility"])
                augmented_source = dict(source_utility)
                augmented_source.update(hybrid_source_utility)
                hybrid_method_checks[method] = method_pass_check(
                    method=hybrid_name, metrics=metrics, risk_metrics=risk_metrics,
                    source_utility=augmented_source, sources=unique_sources,
                    comparator_metrics=comparator_metrics,
                    thresholds=config["decision"]["method_pass"],
                )
        oracle_utility = float(combo_metrics["OracleGate+OracleChoice"]["scalar_utility"])
        choice_error = oracle_utility - float(combo_metrics["OracleGate+LearnedChoice"]["scalar_utility"])
        gate_error = oracle_utility - float(combo_metrics["LearnedGate+OracleChoice"]["scalar_utility"])
        total_error = oracle_utility - float(combo_metrics["LearnedGate+LearnedChoice"]["scalar_utility"])
        interaction = total_error - choice_error - gate_error
        for hybrid, metrics in combo_metrics.items():
            hybrid_rows.append({
                "method": method, "hybrid": hybrid,
                "diagnostic_only": hybrid != "LearnedGate+LearnedChoice",
                "oracle_to_learned_choice_utility_error": choice_error,
                "oracle_to_learned_gate_utility_error": gate_error,
                "total_composed_utility_error": total_error,
                "composition_calibration_interaction": interaction,
                **metrics,
            })

    hybrid_source_rows = _source_metric_rows(
        choices=hybrid_named_choices,
        choice_predictions={}, predicted_a_int={},
        outcomes=outcomes, utility=utility, labels=labels, sources=sources,
        a_int=a_int, target_dr=target_dr, strict_dr=strict_dr,
    )
    for row in hybrid_source_rows:
        row["component"] = "oracle_hybrid"
    source_rows.extend(hybrid_source_rows)

    strict_margins = np.abs(a_dr[strict_dr])
    quartile_probabilities = list(map(float, config["statistics"]["margin"]["quartiles"]))
    quartile_boundaries = np.quantile(strict_margins, quartile_probabilities)
    median_margin = float(quartile_boundaries[1])
    upper_half = strict_dr & (np.abs(a_dr) >= median_margin)
    margin_masks = {
        "Q1": strict_dr & (np.abs(a_dr) <= quartile_boundaries[0]),
        "Q2": strict_dr & (np.abs(a_dr) > quartile_boundaries[0]) & (np.abs(a_dr) <= quartile_boundaries[1]),
        "Q3": strict_dr & (np.abs(a_dr) > quartile_boundaries[1]) & (np.abs(a_dr) <= quartile_boundaries[2]),
        "Q4": strict_dr & (np.abs(a_dr) > quartile_boundaries[2]),
        "upper_half": upper_half,
    }
    margin_rows: list[dict[str, Any]] = []
    upper_choice_metrics: dict[str, dict[str, Any]] = {}
    for method, predicted_choice in choice_prediction_map.items():
        for subset, mask in margin_masks.items():
            metrics = conditional_choice_metrics(predicted_choice, target_dr, sources, mask)
            margin_rows.append({
                "method": method, "margin_subset": subset,
                "lower_q25": float(quartile_boundaries[0]),
                "median_q50": float(quartile_boundaries[1]),
                "upper_q75": float(quartile_boundaries[2]),
                **metrics,
            })
            if method in ADR_METHODS and subset == "upper_half":
                upper_choice_metrics[method] = metrics

    grouped_choices = {**all_composed_choices, **hybrid_named_choices}
    family_rows = _group_metric_rows(
        group_type="family", group_values=families,
        choices=grouped_choices, choice_predictions=choice_prediction_map,
        predicted_a_int=predicted_a_int, outcomes=outcomes, utility=utility,
        labels=labels, sources=sources, a_int=a_int, target_dr=target_dr,
        strict_dr=strict_dr,
    )
    family_rows.extend(_group_metric_rows(
        group_type="horizon", group_values=horizons.astype(str),
        choices=grouped_choices, choice_predictions=choice_prediction_map,
        predicted_a_int=predicted_a_int, outcomes=outcomes, utility=utility,
        labels=labels, sources=sources, a_int=a_int, target_dr=target_dr,
        strict_dr=strict_dr,
    ))
    family_horizon = np.asarray([
        f"{family}|{horizon}" for family, horizon in zip(families, horizons)
    ])
    family_rows.extend(_group_metric_rows(
        group_type="family_horizon", group_values=family_horizon,
        choices=grouped_choices, choice_predictions=choice_prediction_map,
        predicted_a_int=predicted_a_int, outcomes=outcomes, utility=utility,
        labels=labels, sources=sources, a_int=a_int, target_dr=target_dr,
        strict_dr=strict_dr,
    ))
    for row in family_rows:
        if str(row.get("method", "")) in hybrid_named_choices:
            row["component"] = "oracle_hybrid"

    bootstrap_rows = _bootstrap_metric_rows(
        methods=ADR_METHODS, choices=all_composed_choices,
        predicted_a_int=predicted_a_int, predicted_a_dr=predicted_a_dr,
        risk_choice=risk_choice, outcomes=outcomes, utility=utility, labels=labels,
        sources=sources, a_int=a_int, target_dr=target_dr, strict_dr=strict_dr,
        config=config,
    )

    choice_checks = {
        method: choice_pass_check(choice_metrics_lookup[method], config["decision"]["choice_pass"])
        for method in ADR_METHODS
    }
    upper_choice_checks = {
        method: choice_pass_check(upper_choice_metrics[method], config["decision"]["choice_pass"])
        for method in ADR_METHODS
    }
    method_pass_flags = {method: bool(check["passes"]) for method, check in method_checks.items()}
    choice_pass_flags = {method: bool(check["passes"]) for method, check in choice_checks.items()}
    upper_choice_pass_flags = {
        method: bool(check["passes"]) for method, check in upper_choice_checks.items()
    }
    primary_decision = select_primary_decision(
        method_pass=method_pass_flags,
        choice_pass=choice_pass_flags,
        upper_half_choice_pass=upper_choice_pass_flags,
    )
    if primary_decision not in PRIMARY_DECISIONS or primary_decision == "INCONCLUSIVE":
        raise AssertionError("resolver produced a non-exhaustive primary decision")

    secondary_flags: list[str] = []
    if choice_pass_flags["ADR-tiny-nonlinear-choice"] and not any(
        choice_pass_flags[method] for method in LINEAR_METHODS
    ):
        secondary_flags.append("tiny_nonlinear_choice_only")
    if not any(choice_pass_flags.values()) and any(upper_choice_pass_flags.values()):
        secondary_flags.append("upper_half_margin_only_signal")
    gate_thresholds = config["decision"]["gate_diagnostic"]
    for method in ADR_METHODS:
        if not choice_pass_flags[method] or method_pass_flags[method]:
            continue
        if hybrid_method_checks[method]["passes"]:
            secondary_flags.append(f"{method}:composition_failure")
            continue
        gate = gate_lookup[method]
        discrimination_pass = (
            float(gate["balanced_accuracy"]) >= float(gate_thresholds["balanced_accuracy_min"])
            and float(gate["intervention_recall"]) >= float(gate_thresholds["intervention_recall_min"])
            and float(gate["base_recall"]) >= float(gate_thresholds["base_recall_min"])
            and float(gate["auroc"]) >= float(gate_thresholds["auroc_min"])
        )
        secondary_flags.append(
            f"{method}:calibration_failure" if discrimination_pass
            else f"{method}:benefit_gate_failure"
        )

    selected_method = None
    if primary_decision == "METHOD_READY":
        selected_method = max(
            (method for method in ADR_METHODS if method_pass_flags[method]),
            key=lambda method: (
                float(overall_lookup[method]["scalar_utility"]),
                float(overall_lookup[method]["strict_choice_macro_f1"]),
            ),
        )
    decision = {
        "schema_version": 1,
        "phase": "2.5B-R",
        "post_hoc_protocol_amendment": True,
        "primary_decision": primary_decision,
        "selected_method": selected_method,
        "secondary_flags": secondary_flags,
        "data_fingerprint": str(config["inputs"]["data_fingerprint"]),
        "data_fingerprint_sha256": hashlib.sha256(
            json.dumps(
                {key: input_hashes[key] for key in sorted(input_hashes) if key.startswith("capture/")},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "git_commit": _git_commit(),
        "method_pass": method_pass_flags,
        "method_checks": method_checks,
        "choice_pass": choice_pass_flags,
        "choice_checks": choice_checks,
        "upper_half_choice_pass": upper_choice_pass_flags,
        "upper_half_choice_checks": upper_choice_checks,
        "learned_gate_oracle_choice_method_checks": hybrid_method_checks,
        "exact_thresholds": {
            "choice_pass": config["decision"]["choice_pass"],
            "method_pass": config["decision"]["method_pass"],
            "gate_diagnostic": config["decision"]["gate_diagnostic"],
            "intervention_rate_cap": rho,
            "outer_fold_tau": fold_thresholds,
            "margin_quartile_boundaries": list(map(float, quartile_boundaries)),
            "upper_half_margin_rule": config["statistics"]["margin"]["upper_half_rule"],
        },
        "oracle_diagnostics_deployable": False,
        "authorized_next_action": config["decision"]["authorized_next_action"][primary_decision],
    }
    if not isinstance(decision["authorized_next_action"], str) or not decision["authorized_next_action"].strip():
        raise AssertionError("decision must authorize exactly one non-empty next action")

    oof_rows: list[dict[str, Any]] = []
    threshold_by_source = {
        method: {source: fold_thresholds[method][source] for source in unique_sources}
        for method in ADR_METHODS
    }
    for method, choice in grouped_choices.items():
        for index in range(n):
            row = {
                "decision_id": decision_ids[index], "source": sources[index],
                "historical_split": historical_split[index], "family": families[index],
                "horizon": int(horizons[index]), "strict_label": labels[index],
                "strict_advantage": float(strict_advantage[index]),
                "U_B": float(utility[index, 0]), "U_D": float(utility[index, 1]),
                "U_R": float(utility[index, 2]), "A_int": float(a_int[index]),
                "A_DR": float(a_dr[index]), "method": method,
                "choice": OPTIONS[int(choice[index])],
                "selected_outcome": outcomes[index, choice[index]],
                "selected_utility": float(utility[index, choice[index]]),
                "diagnostic_only": (
                    "diagnostic" in method.lower() or method in hybrid_named_choices
                ),
            }
            if method in ADR_METHODS:
                row.update({
                    "predicted_A_int": float(predicted_a_int[method][index]),
                    "predicted_A_DR": float(predicted_a_dr[method][index]),
                    "tau": float(threshold_by_source[method][sources[index]]),
                    "oracle_gate": bool(a_int[index] > 0.0),
                    "oracle_choice": OPTIONS[int(target_dr[index])],
                })
            oof_rows.append(row)

    _write_csv(output_dir / "fold_assignments.csv", fold_rows)
    _write_csv(output_dir / "overall_metrics.csv", overall_rows)
    _write_csv(output_dir / "conditional_choice_metrics.csv", conditional_rows)
    _write_csv(output_dir / "gate_metrics.csv", gate_rows)
    _write_csv(output_dir / "oracle_hybrid_metrics.csv", hybrid_rows)
    _write_csv(output_dir / "source_metrics.csv", source_rows)
    _write_csv(output_dir / "family_metrics.csv", family_rows)
    _write_csv(output_dir / "margin_metrics.csv", margin_rows)
    _write_csv(output_dir / "all_oof_predictions.csv", oof_rows)
    _write_csv(output_dir / "bootstrap_intervals.csv", bootstrap_rows)
    (output_dir / "calibration.json").write_text(
        json.dumps(_native(calibration_records), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "fit_summaries.json").write_text(
        json.dumps(_native(fit_summaries), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "decision.json").write_text(
        json.dumps(_native(decision), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    artifact_names = list(config["outputs"]["required"])
    artifact_names.extend(("fold_assignments.csv", "calibration.json", "fit_summaries.json"))
    artifact_hashes = {
        name: _sha256(output_dir / name)
        for name in artifact_names if name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "kind": "iclr27_advantage_router_resolver",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_dirty(),
        "evidence_level": "exposed-development-only",
        "post_hoc_protocol_amendment": True,
        "new_rollouts": False,
        "fresh_test_outcomes_loaded": False,
        "oracle_diagnostics_deployable": False,
        "data_fingerprint": str(config["inputs"]["data_fingerprint"]),
        "input_sha256": input_hashes,
        "frozen_phase_2_5b_git_commit": frozen_manifest["git_commit"],
        "source_count": len(unique_sources),
        "decision_count": n,
        "outer_folds": len(unique_sources),
        "outer_fit_sources": int(outer["fit_sources"]),
        "outer_calibration_sources": int(outer["calibration_sources"]),
        "models": list(ADR_METHODS),
        "primary_decision": primary_decision,
        "selected_method": selected_method,
        "authorized_next_action": decision["authorized_next_action"],
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "account": os.environ.get("SLURM_JOB_ACCOUNT", "p33100"),
            "partition": os.environ.get("SLURM_JOB_PARTITION", "short"),
        },
        "protocol": {
            "outer_crossfit": outer,
            "pca_scaler_model_fit_role": "outer fit sources only",
            "threshold_selection_role": "five outer-training calibration sources only",
            "outer_test_used_for_selection": False,
            "maximum_source_macro_intervention_rate": rho,
            "margin_diagnostics_used_for_threshold_selection": False,
        },
        "artifact_sha256": artifact_hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_native(manifest), indent=2, sort_keys=True) + "\n"
    )
    return {"manifest": manifest, "decision": decision, "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/iclr27/advantage_router_resolver.yaml",
    )
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--option-support", type=Path)
    parser.add_argument("--support-crossfit", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    result = run_resolver(
        config_path=config_path,
        capture=(args.capture or ROOT / config["inputs"]["capture_path"]).resolve(),
        option_support=(
            args.option_support or ROOT / config["inputs"]["option_support_audit_path"]
        ).resolve(),
        support_crossfit=(
            args.support_crossfit or ROOT / config["inputs"]["support_crossfit_path"]
        ).resolve(),
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "output_dir": result["output_dir"],
        "primary_decision": result["decision"]["primary_decision"],
        "authorized_next_action": result["decision"]["authorized_next_action"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
