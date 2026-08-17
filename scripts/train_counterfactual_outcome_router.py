#!/usr/bin/env python3
"""Freeze the single-frame probabilistic counterfactual outcome router.

The model is deliberately small: train-only PCA-16, robot state and nominal
action, followed by three linear softmax heads (one per option).  Calibration
sources define Base-favoring margins for a predeclared lambda/intervention-rate
frontier; development sources are diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from crashbench.counterfactual_router import (
    OPTIONS,
    OUTCOMES,
    conservative_option_choice,
    option_utilities,
    outcome_probabilities,
)
from scripts.train_minimal_counterfactual_router import (
    _source_weights,
    build_features,
    fit_frame_pca,
    load_capture,
)


DEFAULT_LAMBDAS = (1.0, 2.0, 3.0, 5.0, 8.0)
DEFAULT_TARGET_RATES = tuple(value / 10 for value in range(1, 10))
PRIMARY_TARGET_RATE = 0.4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _standardize(
    x: np.ndarray, sources: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = _source_weights(sources)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance) + 1e-6
    return (x - mean) / scale, mean, scale


def fit_outcome_softmax(
    x: np.ndarray,
    outcomes: np.ndarray,
    sources: np.ndarray,
    *,
    l2: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit three source-balanced linear multinomial outcome heads."""

    xs, mean, scale = _standardize(np.asarray(x, dtype=np.float64), sources)
    design = np.column_stack((xs, np.ones(len(xs))))
    label_index = {name: index for index, name in enumerate(OUTCOMES)}
    labels = np.asarray([
        [label_index[str(value)] for value in row] for row in outcomes
    ], dtype=np.int64)
    weights = _source_weights(sources)
    normalizer = float(weights.sum() * len(OPTIONS))
    rows = np.arange(len(labels))[:, None]
    options = np.arange(len(OPTIONS))[None, :]

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coef = flat.reshape(design.shape[1], len(OPTIONS), len(OUTCOMES))
        logits = np.einsum("nd,dok->nok", design, coef)
        probabilities = outcome_probabilities(logits)
        likelihood = probabilities[rows, options, labels]
        loss = -float(np.sum(weights[:, None] * np.log(likelihood + 1e-12))) / normalizer
        loss += 0.5 * float(l2) * float(np.sum(coef[:-1] ** 2))
        residual = probabilities.copy()
        residual[rows, options, labels] -= 1.0
        residual *= weights[:, None, None] / normalizer
        gradient = np.einsum("nd,nok->dok", design, residual)
        gradient[:-1] += float(l2) * coef[:-1]
        return loss, gradient.ravel()

    initial = np.zeros(design.shape[1] * len(OPTIONS) * len(OUTCOMES))
    fitted = minimize(
        objective, initial, method="L-BFGS-B", jac=True,
        options={"maxiter": 500, "ftol": 1e-11},
    )
    coef = fitted.x.reshape(design.shape[1], len(OPTIONS), len(OUTCOMES))
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "outcome_coef": coef,
    }, {
        "converged": bool(fitted.success),
        "iterations": int(fitted.nit),
        "weighted_cross_entropy_with_l2": float(fitted.fun),
    }


def fit_risk_logistic(
    x: np.ndarray,
    target: np.ndarray,
    sources: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the binary Base-catastrophe baseline on the identical features."""

    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    target = np.asarray(target, dtype=np.float64)
    weights = _source_weights(sources)
    normalizer = float(weights.sum())

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ coef
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        loss = -float(np.sum(weights * (
            target * np.log(probabilities + 1e-12)
            + (1.0 - target) * np.log(1.0 - probabilities + 1e-12)
        ))) / normalizer
        loss += 0.5 * float(l2) * float(np.sum(coef[:-1] ** 2))
        gradient = design.T @ (weights * (probabilities - target)) / normalizer
        gradient[:-1] += float(l2) * coef[:-1]
        return loss, gradient

    fitted = minimize(
        objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
        options={"maxiter": 500, "ftol": 1e-11},
    )
    return fitted.x, {
        "converged": bool(fitted.success),
        "iterations": int(fitted.nit),
        "weighted_cross_entropy_with_l2": float(fitted.fun),
    }


def predict_model(model: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((
        (np.asarray(x, dtype=np.float64) - model["feature_mean"])
        / model["feature_scale"],
        np.ones(len(x)),
    ))
    probabilities = outcome_probabilities(
        np.einsum("nd,dok->nok", design, model["outcome_coef"])
    )
    risk_logits = design @ model["risk_coef"]
    risk = 1.0 / (1.0 + np.exp(-np.clip(risk_logits, -40, 40)))
    return probabilities, risk


def realized_utility(outcomes: np.ndarray, catastrophe_cost: float) -> np.ndarray:
    return np.where(
        outcomes == "task_success", 1.0,
        np.where(outcomes == "catastrophe", -float(catastrophe_cost), 0.0),
    ).astype(np.float64)


def _candidate_margins(probabilities: np.ndarray, catastrophe_cost: float) -> list[float]:
    utility = option_utilities(probabilities, catastrophe_cost)
    intervention = 1 + np.argmax(utility[:, 1:], axis=1)
    advantage = utility[np.arange(len(utility)), intervention] - utility[:, 0]
    positive = sorted({float(value) for value in advantage if value >= 0.0})
    candidates = [0.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(positive, positive[1:]))
    candidates.append((positive[-1] + 1e-6) if positive else 1e-6)
    return sorted(set(candidates))


def calibrate_margin_for_rate(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    sources: np.ndarray,
    *,
    catastrophe_cost: float,
    target_rate: float,
) -> dict[str, float]:
    weights = _source_weights(sources)
    utility = realized_utility(outcomes, catastrophe_cost)
    candidates = []
    for margin in _candidate_margins(probabilities, catastrophe_cost):
        choice = conservative_option_choice(
            probabilities, catastrophe_cost=catastrophe_cost,
            intervention_margin=margin,
        )
        rate = float(np.average(choice != 0, weights=weights))
        value = float(np.average(utility[np.arange(len(choice)), choice], weights=weights))
        candidates.append((-abs(rate - target_rate), value, -rate, margin, rate))
    best = max(candidates)
    return {
        "delta": float(best[-2]),
        "calibration_intervention_rate": float(best[-1]),
        "calibration_source_balanced_utility": float(best[1]),
        "target_intervention_rate": float(target_rate),
    }


def calibrate_risk_threshold_for_rate(
    scores: np.ndarray, sources: np.ndarray, target_rate: float
) -> dict[str, float]:
    unique = sorted({float(value) for value in scores})
    thresholds = [-1e-6, 1.000001]
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    weights = _source_weights(sources)
    candidates = []
    for threshold in thresholds:
        rate = float(np.average(scores > threshold, weights=weights))
        candidates.append((-abs(rate - target_rate), -rate, threshold, rate))
    best = max(candidates)
    return {
        "threshold": float(best[-2]),
        "calibration_intervention_rate": float(best[-1]),
        "target_intervention_rate": float(target_rate),
    }


def evaluate_choices(
    choice: np.ndarray, outcomes: np.ndarray, sources: np.ndarray, catastrophe_cost: float
) -> dict[str, Any]:
    selected = outcomes[np.arange(len(choice)), choice]
    utility = realized_utility(outcomes, catastrophe_cost)
    selected_utility = utility[np.arange(len(choice)), choice]
    source_values = [
        float(selected_utility[sources == source].mean())
        for source in sorted(set(sources))
    ]
    counts = Counter(str(value) for value in selected)
    return {
        "decisions": int(len(choice)),
        "choice_counts": {
            option: int(np.sum(choice == index)) for index, option in enumerate(OPTIONS)
        },
        "rates": {name: counts[name] / len(choice) for name in OUTCOMES},
        "intervention_rate": float(np.mean(choice != 0)),
        "mean_utility": float(np.mean(selected_utility)),
        "source_macro_mean_utility": float(np.mean(source_values)),
    }


def train_and_freeze(
    capture: Path,
    output: Path,
    *,
    pca_components: int,
    l2: float,
) -> dict[str, Any]:
    data = load_capture(capture, catastrophe_cost=5.0)
    train = data["splits"] == "train"
    calibration = data["splits"] == "calibration"
    development = data["splits"] == "development"
    pca = fit_frame_pca(
        data["arrays"]["hidden"], data["arrays"]["history_mask"], train,
        pca_components,
    )
    features = build_features(
        data["arrays"], pca, history=False, include_robot_action=True,
    )
    model, outcome_fit = fit_outcome_softmax(
        features[train], data["outcomes"][train], data["sources"][train], l2=l2,
    )
    base_catastrophe = (data["outcomes"][:, 0] == "catastrophe").astype(np.float64)
    risk_coef, risk_fit = fit_risk_logistic(
        features[train], base_catastrophe[train], data["sources"][train],
        mean=model["feature_mean"], scale=model["feature_scale"], l2=l2,
    )
    model["risk_coef"] = risk_coef
    probabilities, risk_scores = predict_model(model, features)

    frontier = {}
    for catastrophe_cost in DEFAULT_LAMBDAS:
        key = f"lambda_{catastrophe_cost:g}"
        frontier[key] = {
            f"target_{target_rate:.1f}": calibrate_margin_for_rate(
                probabilities[calibration], data["outcomes"][calibration],
                data["sources"][calibration], catastrophe_cost=catastrophe_cost,
                target_rate=target_rate,
            )
            for target_rate in DEFAULT_TARGET_RATES
        }
    risk_frontier = {
        f"target_{target_rate:.1f}": calibrate_risk_threshold_for_rate(
            risk_scores[calibration], data["sources"][calibration], target_rate,
        )
        for target_rate in DEFAULT_TARGET_RATES
    }

    development_diagnostic = {}
    for catastrophe_cost in DEFAULT_LAMBDAS:
        point = frontier[f"lambda_{catastrophe_cost:g}"][
            f"target_{PRIMARY_TARGET_RATE:.1f}"
        ]
        choice = conservative_option_choice(
            probabilities[development], catastrophe_cost=catastrophe_cost,
            intervention_margin=point["delta"],
        )
        development_diagnostic[f"lambda_{catastrophe_cost:g}"] = evaluate_choices(
            choice, data["outcomes"][development], data["sources"][development],
            catastrophe_cost,
        )
    risk_point = risk_frontier[f"target_{PRIMARY_TARGET_RATE:.1f}"]
    risk_choice = np.where(risk_scores[development] > risk_point["threshold"], 2, 0)
    development_diagnostic["binary_risk_retreat"] = evaluate_choices(
        risk_choice, data["outcomes"][development], data["sources"][development], 5.0,
    )

    artifact = output.with_suffix(".npz")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        pca_mean=pca[0],
        pca_components=pca[1],
        feature_mean=model["feature_mean"],
        feature_scale=model["feature_scale"],
        outcome_coef=model["outcome_coef"],
        risk_coef=model["risk_coef"],
    )
    capture_files = (
        "capture_manifest.json", "decision_metadata.json",
        "option_rollouts.jsonl", "decision_features.npz",
    )
    manifest = {
        "schema_version": 1,
        "kind": "frozen_counterfactual_outcome_router",
        "artifact_npz": artifact.name,
        "artifact_npz_sha256": _sha256(artifact),
        "capture_root": str(capture),
        "capture_sha256": {name: _sha256(capture / name) for name in capture_files},
        "options": list(OPTIONS),
        "outcomes": list(OUTCOMES),
        "feature_contract": {
            "history": "single_frame",
            "inputs": "hidden_robot_action",
            "pca_components": int(pca[1].shape[1]),
            "feature_dimension": int(features.shape[1]),
        },
        "training": {
            "model": "three source-balanced linear softmax outcome heads",
            "risk_baseline": "source-balanced binary logistic Base-catastrophe head",
            "l2": float(l2),
            "source_counts": {
                split: len(set(data["sources"][data["splits"] == split]))
                for split in ("train", "calibration", "development")
            },
            "decision_counts": {
                split: int(np.sum(data["splits"] == split))
                for split in ("train", "calibration", "development")
            },
            "outcome_fit": outcome_fit,
            "risk_fit": risk_fit,
        },
        "calibration": {
            "lambdas": list(DEFAULT_LAMBDAS),
            "target_intervention_rates": list(DEFAULT_TARGET_RATES),
            "primary_target_intervention_rate": PRIMARY_TARGET_RATE,
            "router_frontier": frontier,
            "binary_risk_retreat_frontier": risk_frontier,
        },
        "development_diagnostic_only": development_diagnostic,
        "all_capture_source_state_sha256": sorted(set(map(str, data["sources"]))),
        "protocol_note": (
            "development outcomes are reported diagnostically but never fit the PCA, "
            "outcome heads, risk head, margins, or thresholds"
        ),
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()
    result = train_and_freeze(
        args.capture.resolve(), args.output.resolve(),
        pca_components=args.pca_components, l2=args.l2,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "artifact": result["artifact_npz"],
        "development_diagnostic_only": result["development_diagnostic_only"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
