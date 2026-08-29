#!/usr/bin/env python3
"""Train the source-disjoint ICLR 2027 router baseline suite.

This entry point intentionally consumes only the existing train/calibration/
development exact-state capture.  All learned methods share one train-only
PCA and source-balanced fitting; development outcomes are written for the
separate analysis step but are never used to fit a model or operating point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
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

from crashbench.counterfactual_router import OPTIONS, OUTCOMES, outcome_probabilities
from scripts.train_minimal_counterfactual_router import (
    _source_weights,
    build_features,
    fit_frame_pca,
    fit_weighted_ridge,
    load_capture,
    predict_ridge,
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


def assert_source_disjoint(sources: Sequence[str], splits: Sequence[str]) -> None:
    """Require every source state to occur in exactly one split."""

    allocation: dict[str, str] = {}
    for source, split in zip(map(str, sources), map(str, splits)):
        previous = allocation.setdefault(source, split)
        if previous != split:
            raise ValueError(f"source {source} leaks across {previous}/{split}")


def source_balanced_mean(values: np.ndarray, sources: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.average(values, weights=_source_weights(np.asarray(sources))))


def realized_utility(
    outcomes: np.ndarray, catastrophe_cost: float, noncompletion_cost: float = 0.0
) -> np.ndarray:
    """Map exhaustive outcomes to the frozen scalar preference."""

    return np.where(
        outcomes == "task_success",
        1.0,
        np.where(
            outcomes == "catastrophe",
            -float(catastrophe_cost),
            -float(noncompletion_cost),
        ),
    ).astype(np.float64)


def expected_utility(
    probabilities: np.ndarray, catastrophe_cost: float, noncompletion_cost: float = 0.0
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    return (
        values[..., 0]
        - float(catastrophe_cost) * values[..., 1]
        - float(noncompletion_cost) * values[..., 2]
    )


def oracle_choice_base_favoring(
    utility: np.ndarray, option_costs: Sequence[float]
) -> np.ndarray:
    """Choose realized-best options with Base/cost/order tie breaking."""

    utility = np.asarray(utility, dtype=np.float64)
    costs = np.asarray(option_costs, dtype=np.float64)
    if utility.ndim != 2 or utility.shape[1] != len(costs):
        raise ValueError("utility and option costs disagree")
    choices = np.empty(len(utility), dtype=np.int64)
    for index, row in enumerate(utility):
        candidates = np.flatnonzero(np.isclose(row, row.max(), atol=1e-12, rtol=0.0))
        if 0 in candidates:
            choices[index] = 0
            continue
        candidate_costs = costs[candidates]
        choices[index] = int(candidates[np.argmin(candidate_costs)])
    return choices


def _intervention_choice_cost_favoring(
    utility: np.ndarray, intervention_costs: Sequence[float]
) -> np.ndarray:
    """Resolve a Detour/Retreat tie by cost and then fixed order."""

    utility = np.asarray(utility, dtype=np.float64)
    costs = np.asarray(intervention_costs, dtype=np.float64)
    choices = np.empty(len(utility), dtype=np.int64)
    for index, row in enumerate(utility):
        candidates = np.flatnonzero(np.isclose(row, row.max(), atol=1e-12, rtol=0.0))
        choices[index] = int(candidates[np.argmin(costs[candidates])])
    return choices


def _candidate_thresholds(scores: np.ndarray) -> list[float]:
    values = sorted({float(value) for value in np.asarray(scores).reshape(-1)})
    if not values:
        raise ValueError("cannot calibrate an empty score array")
    return [
        float("-inf"),
        *((left + right) / 2.0 for left, right in zip(values, values[1:])),
        float("inf"),
    ]


def calibrate_rate_threshold(
    scores: np.ndarray,
    sources: np.ndarray,
    target_rate: float,
    minimum_threshold: float | None = None,
) -> dict[str, float]:
    """Match a source-balanced intervention rate with a strict ``>`` rule."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    sources = np.asarray(sources)
    weights = _source_weights(sources)
    candidates = []
    thresholds = _candidate_thresholds(scores)
    if minimum_threshold is not None:
        thresholds = sorted({
            float(minimum_threshold),
            *(value for value in thresholds if value >= float(minimum_threshold)),
        })
    for threshold in thresholds:
        rate = float(np.average(scores > threshold, weights=weights))
        # Prefer the lower rate when equally close, then the larger threshold.
        candidates.append((-abs(rate - float(target_rate)), -rate, threshold, rate))
    best = max(candidates, key=lambda row: row[:3])
    return {
        "threshold": float(best[2]),
        "source_balanced_rate": float(best[3]),
        "target_rate": float(target_rate),
    }


def _utility_from_argument(
    outcomes: np.ndarray, utility_values: Mapping[str, float] | np.ndarray
) -> np.ndarray:
    if isinstance(utility_values, Mapping):
        return np.vectorize(lambda value: float(utility_values[str(value)]))(outcomes)
    utility = np.asarray(utility_values, dtype=np.float64)
    if utility.shape != outcomes.shape:
        raise ValueError("utility_values must be an outcome map or an N x 3 matrix")
    return utility


def select_best_fixed(
    risk_scores: np.ndarray,
    outcomes: np.ndarray,
    sources: np.ndarray,
    threshold: float,
    utility_values: Mapping[str, float] | np.ndarray,
) -> dict[str, Any]:
    """Freeze Detour or Retreat using calibration-only balanced utility."""

    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    outcomes = np.asarray(outcomes)
    sources = np.asarray(sources)
    utility = _utility_from_argument(outcomes, utility_values)
    candidates = []
    for option in (1, 2):
        choice = np.where(risk_scores > float(threshold), option, 0)
        value = source_balanced_mean(utility[np.arange(len(choice)), choice], sources)
        candidates.append((value, -option, option))
    best = max(candidates)
    return {
        "option_index": int(best[2]),
        "option": OPTIONS[int(best[2])],
        "source_balanced_utility": float(best[0]),
        "threshold": float(threshold),
    }


def _standardize(
    x: np.ndarray, sources: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = _source_weights(sources)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance) + 1e-6
    return (x - mean) / scale, mean, scale


def fit_multinomial(
    x: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    classes: int,
    l2: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit a compact source-balanced multinomial logistic model."""

    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    xs, mean, scale = _standardize(x, sources)
    design = np.column_stack((xs, np.ones(len(xs))))
    weights = _source_weights(sources)
    rows = np.arange(len(labels))
    normalizer = float(weights.sum())

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coef = flat.reshape(design.shape[1], classes)
        probabilities = outcome_probabilities(design @ coef)
        loss = -float(np.sum(weights * np.log(probabilities[rows, labels] + 1e-12)))
        loss /= normalizer
        # Regularizing the intercept keeps a finite optimum when a small split
        # contains no example of one class (notably Retreat task-success).
        loss += 0.5 * float(l2) * float(np.sum(coef ** 2))
        residual = probabilities.copy()
        residual[rows, labels] -= 1.0
        residual *= weights[:, None] / normalizer
        gradient = design.T @ residual
        gradient += float(l2) * coef
        return loss, gradient.ravel()

    fitted = minimize(
        objective,
        np.zeros(design.shape[1] * classes),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-11},
    )
    return {
        "mean": mean,
        "scale": scale,
        "coef": fitted.x.reshape(design.shape[1], classes),
    }, {
        "converged": bool(fitted.success),
        "iterations": int(fitted.nit),
        "weighted_cross_entropy_with_l2": float(fitted.fun),
    }


def predict_multinomial(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    design = np.column_stack((
        (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"],
        np.ones(len(x)),
    ))
    return outcome_probabilities(design @ model["coef"])


def _require_converged(name: str, summary: Mapping[str, Any]) -> None:
    if not bool(summary.get("converged", False)):
        raise RuntimeError(f"required baseline fit did not converge: {name}")


def fit_outcome_heads(
    x: np.ndarray,
    outcomes: np.ndarray,
    sources: np.ndarray,
    *,
    l2: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit three finite, source-balanced option-outcome softmax heads."""

    label_index = {name: index for index, name in enumerate(OUTCOMES)}
    labels = np.asarray([
        [label_index[str(value)] for value in row] for row in outcomes
    ], dtype=np.int64)
    fitted_models = []
    summaries: dict[str, Any] = {}
    for option_index, option in enumerate(OPTIONS):
        model, summary = fit_multinomial(
            x, labels[:, option_index], sources, classes=len(OUTCOMES), l2=l2
        )
        _require_converged(f"outcome head {option}", summary)
        fitted_models.append(model)
        summaries[option] = summary
    return {
        "feature_mean": fitted_models[0]["mean"],
        "feature_scale": fitted_models[0]["scale"],
        "outcome_coef": np.stack(
            [model["coef"] for model in fitted_models], axis=1
        ),
    }, {
        "converged": True,
        "regularization": "l2 including intercept",
        "per_option": summaries,
    }


def fit_binary_logistic(
    x: np.ndarray, target: np.ndarray, sources: np.ndarray, *, l2: float
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model, summary = fit_multinomial(
        x,
        np.asarray(target, dtype=np.int64),
        sources,
        classes=2,
        l2=l2,
    )
    return model, summary


def predict_outcome_model(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    design = np.column_stack((
        (np.asarray(x, dtype=np.float64) - model["feature_mean"])
        / model["feature_scale"],
        np.ones(len(x)),
    ))
    return outcome_probabilities(np.einsum("nd,dok->nok", design, model["outcome_coef"]))


def _choice_from_values(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    intervention = 1 + np.argmax(values[:, 1:], axis=1)
    advantage = values[np.arange(len(values)), intervention] - values[:, 0]
    return np.where(advantage > float(threshold), intervention, 0).astype(np.int64)


def _advantage_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.max(values[:, 1:] - values[:, [0]], axis=1)


def _choice_from_direct_probabilities(
    probabilities: np.ndarray, threshold: float
) -> np.ndarray:
    intervention = 1 + np.argmax(probabilities[:, 1:], axis=1)
    score = probabilities[np.arange(len(probabilities)), intervention] - probabilities[:, 0]
    return np.where(score > float(threshold), intervention, 0).astype(np.int64)


def _latency_us(choice_fn: Callable[[], np.ndarray], decisions: int) -> float:
    repeats = max(20, min(200, 20000 // max(1, decisions)))
    choice_fn()
    start = time.perf_counter()
    for _ in range(repeats):
        choice_fn()
    elapsed = time.perf_counter() - start
    return float(elapsed * 1e6 / (repeats * max(1, decisions)))


class MethodTable:
    """Collect rectangular prediction arrays and paper-facing metadata."""

    def __init__(self, decisions: int) -> None:
        self.decisions = decisions
        self.names: list[str] = []
        self.choices: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.outcome_probabilities: list[np.ndarray] = []
        self.choice_probabilities: list[np.ndarray] = []
        self.parameter_counts: list[int] = []
        self.latencies: list[float] = []
        self.metadata: dict[str, dict[str, Any]] = {}

    def add(
        self,
        name: str,
        choice: np.ndarray,
        *,
        predicted_values: np.ndarray | None = None,
        outcome_probabilities_: np.ndarray | None = None,
        choice_probabilities_: np.ndarray | None = None,
        parameter_count: int = 0,
        latency_us: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        n = self.decisions
        choice = np.asarray(choice, dtype=np.int16)
        if choice.shape != (n,):
            raise ValueError(f"{name} returned choices with shape {choice.shape}")
        nan_values = np.full((n, len(OPTIONS)), np.nan, dtype=np.float64)
        nan_outcomes = np.full(
            (n, len(OPTIONS), len(OUTCOMES)), np.nan, dtype=np.float64
        )
        self.names.append(name)
        self.choices.append(choice)
        self.values.append(
            nan_values if predicted_values is None else np.asarray(predicted_values)
        )
        self.outcome_probabilities.append(
            nan_outcomes
            if outcome_probabilities_ is None
            else np.asarray(outcome_probabilities_)
        )
        self.choice_probabilities.append(
            nan_values
            if choice_probabilities_ is None
            else np.asarray(choice_probabilities_)
        )
        self.parameter_counts.append(int(parameter_count))
        self.latencies.append(float(latency_us))
        self.metadata[name] = dict(metadata or {})


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("baseline suite config must be a mapping")
    return payload


def _preflight_capture_splits(capture: Path, config: Mapping[str, Any]) -> set[str]:
    """Validate split roles before any option-outcome file is opened."""

    metadata_path = capture / "decision_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    observed = {str(row["split"]) for row in metadata}
    allowed = set(map(str, config["data_policy"]["allowed_splits"]))
    disallowed = observed - allowed
    if disallowed:
        raise ValueError(
            "phase-2 trainer refuses outcome-bearing test splits: "
            + ", ".join(sorted(disallowed))
        )
    required = {"train", "calibration", "development"}
    if not required <= observed:
        raise ValueError(f"capture lacks required splits: {sorted(required - observed)}")
    assert_source_disjoint(
        [str(row["source_state_sha256"]) for row in metadata],
        [str(row["split"]) for row in metadata],
    )
    return observed


def train_suite(capture: Path, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Fit every baseline and write a complete prediction artifact."""

    capture = Path(capture).resolve()
    output_dir = Path(output_dir).resolve()
    observed_splits = _preflight_capture_splits(capture, config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = config["protocol"]
    primary_lambda = float(protocol["primary_lambda"])
    eta = float(protocol.get("noncompletion_cost", 0.0))
    target_rate = float(protocol["primary_target_intervention_rate"])
    l2 = float(config["model"]["logistic_l2"])
    ridge_alpha = float(config["model"]["ridge_alpha"])
    pca_components = int(config["features"]["pca_components"])
    option_costs = [float(value) for value in protocol["option_costs"]]

    data = load_capture(capture, primary_lambda)
    assert_source_disjoint(data["sources"], data["splits"])
    if observed_splits != set(map(str, data["splits"])):
        raise ValueError("decision metadata split roles changed while loading capture")

    train = data["splits"] == "train"
    calibration = data["splits"] == "calibration"
    n = len(data["outcomes"])
    utility = realized_utility(data["outcomes"], primary_lambda, eta)
    pca = fit_frame_pca(
        data["arrays"]["hidden"],
        data["arrays"]["history_mask"],
        train,
        pca_components,
        seed=int(config["model"].get("seed", 2027)),
    )
    x_full = build_features(
        data["arrays"], pca, history=False, include_robot_action=True
    )
    hidden = np.asarray(data["arrays"]["hidden"], dtype=np.float64)[:, -1]
    x_hidden = (hidden - pca[0]) @ pca[1]
    x_state_action = np.concatenate((
        np.asarray(data["arrays"]["robot_state"], dtype=np.float64)[:, -1],
        np.asarray(data["arrays"]["nominal_action"], dtype=np.float64)[:, -1],
    ), axis=1)

    methods = MethodTable(n)
    model_arrays: dict[str, np.ndarray] = {
        "pca_mean": pca[0],
        "pca_components": pca[1],
    }
    methods.add("Base", np.zeros(n, dtype=np.int64), metadata={"family": "fixed"})
    methods.add(
        "Always Detour", np.ones(n, dtype=np.int64), metadata={"family": "fixed"}
    )
    methods.add(
        "Always Retreat", np.full(n, 2, dtype=np.int64), metadata={"family": "fixed"}
    )

    fit_summaries: dict[str, Any] = {}
    calibration_records: dict[str, Any] = {}

    # One scalar Base-catastrophe model and one rate threshold are shared by B3--B6.
    base_catastrophe = (data["outcomes"][:, 0] == "catastrophe").astype(np.int64)
    risk_model, risk_fit = fit_binary_logistic(
        x_full[train], base_catastrophe[train], data["sources"][train], l2=l2
    )
    _require_converged("binary Base-catastrophe risk", risk_fit)
    risk_probability = predict_multinomial(risk_model, x_full)[:, 1]
    model_arrays.update({
        "risk_mean": risk_model["mean"],
        "risk_scale": risk_model["scale"],
        "risk_coef": risk_model["coef"],
    })
    risk_cal = calibrate_rate_threshold(
        risk_probability[calibration], data["sources"][calibration], target_rate
    )
    risk_threshold = float(risk_cal["threshold"])
    risk_gate = risk_probability > risk_threshold
    risk_detour_prob = np.column_stack((
        1.0 - risk_probability, risk_probability, np.zeros(n)
    ))
    risk_retreat_prob = np.column_stack((
        1.0 - risk_probability, np.zeros(n), risk_probability
    ))
    risk_params = int(risk_model["coef"].size)
    risk_latency = _latency_us(
        lambda: np.where(
            predict_multinomial(risk_model, x_full)[:, 1] > risk_threshold, 2, 0
        ), n
    )
    methods.add(
        "Risk->Retreat",
        np.where(risk_gate, 2, 0),
        choice_probabilities_=risk_retreat_prob,
        parameter_count=risk_params,
        latency_us=risk_latency,
        metadata={"family": "scalar_risk", "fixed_option": OPTIONS[2]},
    )
    methods.add(
        "Risk->Detour",
        np.where(risk_gate, 1, 0),
        choice_probabilities_=risk_detour_prob,
        parameter_count=risk_params,
        latency_us=risk_latency,
        metadata={"family": "scalar_risk", "fixed_option": OPTIONS[1]},
    )
    best_fixed = select_best_fixed(
        risk_probability[calibration],
        data["outcomes"][calibration],
        data["sources"][calibration],
        risk_threshold,
        utility[calibration],
    )
    methods.add(
        "Risk->BestFixed",
        np.where(risk_gate, int(best_fixed["option_index"]), 0),
        choice_probabilities_=(
            risk_detour_prob
            if int(best_fixed["option_index"]) == 1
            else risk_retreat_prob
        ),
        parameter_count=risk_params,
        latency_us=risk_latency,
        metadata={
            "family": "scalar_risk",
            "fixed_option": best_fixed["option"],
            "selected_on": "calibration",
        },
    )
    fit_summaries["risk"] = risk_fit
    calibration_records["shared_risk_rate"] = risk_cal
    calibration_records["best_fixed"] = best_fixed

    # B6: the gate remains scalar risk; the second stage learns only which
    # intervention is useful on beneficial training decisions.
    intervention_utility = utility[:, 1:]
    two_stage_target = 1 + _intervention_choice_cost_favoring(
        intervention_utility, option_costs[1:]
    )
    beneficial = intervention_utility.max(axis=1) > utility[:, 0] + 1e-12
    two_stage_train = train & beneficial
    option_model, option_fit = fit_multinomial(
        x_full[two_stage_train],
        two_stage_target[two_stage_train] - 1,
        data["sources"][two_stage_train],
        classes=2,
        l2=l2,
    )
    _require_converged("two-stage intervention option", option_fit)
    option_probability = predict_multinomial(option_model, x_full)
    model_arrays.update({
        "two_stage_mean": option_model["mean"],
        "two_stage_scale": option_model["scale"],
        "two_stage_coef": option_model["coef"],
    })
    option_choice = 1 + np.argmax(option_probability, axis=1)
    two_stage_choice = np.where(risk_gate, option_choice, 0)
    two_stage_probs = np.column_stack((
        1.0 - risk_probability,
        risk_probability * option_probability[:, 0],
        risk_probability * option_probability[:, 1],
    ))
    methods.add(
        "Risk+TwoStage",
        two_stage_choice,
        choice_probabilities_=two_stage_probs,
        parameter_count=risk_params + int(option_model["coef"].size),
        latency_us=_latency_us(
            lambda: np.where(
                predict_multinomial(risk_model, x_full)[:, 1] > risk_threshold,
                1 + np.argmax(predict_multinomial(option_model, x_full), axis=1),
                0,
            ), n
        ),
        metadata={
            "family": "two_stage",
            "option_training_decisions": int(two_stage_train.sum()),
            "option_training_rule": "train-only beneficial intervention",
        },
    )
    fit_summaries["two_stage_option"] = option_fit

    # B7: direct realized-best choice, with the frozen Base/cost/order tie rule.
    direct_labels = oracle_choice_base_favoring(utility, option_costs)
    direct_model, direct_fit = fit_multinomial(
        x_full[train], direct_labels[train], data["sources"][train], classes=3, l2=l2
    )
    _require_converged("direct realized-choice", direct_fit)
    direct_probability = predict_multinomial(direct_model, x_full)
    model_arrays.update({
        "direct_choice_mean": direct_model["mean"],
        "direct_choice_scale": direct_model["scale"],
        "direct_choice_coef": direct_model["coef"],
    })
    direct_candidate = 1 + np.argmax(direct_probability[:, 1:], axis=1)
    direct_score = (
        direct_probability[np.arange(n), direct_candidate] - direct_probability[:, 0]
    )
    direct_cal = calibrate_rate_threshold(
        direct_score[calibration], data["sources"][calibration], target_rate,
    )
    direct_threshold = float(direct_cal["threshold"])
    direct_choice = _choice_from_direct_probabilities(
        direct_probability, direct_threshold
    )
    methods.add(
        "DirectChoice",
        direct_choice,
        choice_probabilities_=direct_probability,
        parameter_count=int(direct_model["coef"].size),
        latency_us=_latency_us(
            lambda: _choice_from_direct_probabilities(
                predict_multinomial(direct_model, x_full), direct_threshold
            ), n
        ),
        metadata={"family": "direct_choice", "threshold": direct_threshold},
    )
    fit_summaries["direct_choice"] = direct_fit
    calibration_records["direct_choice"] = direct_cal

    # B8: primary single-lambda Direct-Q and separately fitted Q heads for every
    # declared preference.  Only the primary model enters the hard gate.
    lambda_grid = [float(value) for value in protocol["lambda_grid"]]
    direct_q_models: dict[str, Any] = {}
    for catastrophe_cost in lambda_grid:
        name = (
            "DirectQ" if np.isclose(catastrophe_cost, primary_lambda)
            else f"DirectQ-lambda{catastrophe_cost:g}"
        )
        q_target = realized_utility(data["outcomes"], catastrophe_cost, eta)
        q_model = fit_weighted_ridge(
            x_full[train], q_target[train], data["sources"][train], ridge_alpha
        )
        q_prediction = predict_ridge(q_model, x_full)
        q_key = f"direct_q_lambda_{catastrophe_cost:g}".replace(".", "p")
        model_arrays.update({
            f"{q_key}_mean": q_model["mean"],
            f"{q_key}_scale": q_model["scale"],
            f"{q_key}_coef": q_model["coef"],
        })
        q_score = _advantage_score(q_prediction)
        q_cal = calibrate_rate_threshold(
            q_score[calibration], data["sources"][calibration], target_rate,
        )
        q_threshold = float(q_cal["threshold"])
        q_choice = _choice_from_values(q_prediction, q_threshold)
        methods.add(
            name,
            q_choice,
            predicted_values=q_prediction,
            parameter_count=int(q_model["coef"].size),
            latency_us=_latency_us(
                lambda qm=q_model, qt=q_threshold: _choice_from_values(
                    predict_ridge(qm, x_full), qt
                ), n
            ),
            metadata={
                "family": "direct_q",
                "training_lambda": catastrophe_cost,
                "primary_gate_method": bool(np.isclose(catastrophe_cost, primary_lambda)),
                "threshold": q_threshold,
            },
        )
        calibration_records[name] = q_cal
        direct_q_models[name] = {
            "training_lambda": catastrophe_cost,
            "parameter_count": int(q_model["coef"].size),
        }

    # B9: directly regress Base-relative intervention advantages.
    advantage_target = utility[:, 1:] - utility[:, [0]]
    advantage_model = fit_weighted_ridge(
        x_full[train], advantage_target[train], data["sources"][train], ridge_alpha
    )
    predicted_advantage = predict_ridge(advantage_model, x_full)
    model_arrays.update({
        "pairwise_mean": advantage_model["mean"],
        "pairwise_scale": advantage_model["scale"],
        "pairwise_coef": advantage_model["coef"],
    })
    pairwise_values = np.column_stack((np.zeros(n), predicted_advantage))
    pairwise_score = np.max(predicted_advantage, axis=1)
    pairwise_cal = calibrate_rate_threshold(
        pairwise_score[calibration], data["sources"][calibration], target_rate,
    )
    pairwise_threshold = float(pairwise_cal["threshold"])
    pairwise_choice = _choice_from_values(pairwise_values, pairwise_threshold)
    methods.add(
        "PairwiseAdvantage",
        pairwise_choice,
        predicted_values=pairwise_values,
        parameter_count=int(advantage_model["coef"].size),
        latency_us=_latency_us(
            lambda: _choice_from_values(
                np.column_stack((
                    np.zeros(n), predict_ridge(advantage_model, x_full)
                )),
                pairwise_threshold,
            ), n
        ),
        metadata={
            "family": "pairwise_advantage",
            "rate_matched_advantage_threshold": pairwise_threshold,
            "threshold_is_not_a_physical_cost": True,
        },
    )
    calibration_records["PairwiseAdvantage"] = pairwise_cal

    # B10--B12: identical outcome heads over full, hidden-only, and deployable
    # state/action-only features.  The latter two are representation ablations.
    outcome_variants = {
        "OutcomeRouter": (x_full, "hidden_robot_action"),
        "OutcomeRouter-hidden-only": (x_hidden, "hidden_only"),
        "OutcomeRouter-state-action-only": (x_state_action, "state_action_only"),
    }
    for name, (features, feature_name) in outcome_variants.items():
        outcome_model, outcome_fit = fit_outcome_heads(
            features[train],
            data["outcomes"][train],
            data["sources"][train],
            l2=l2,
        )
        probabilities = predict_outcome_model(outcome_model, features)
        outcome_key = name.lower().replace("-", "_")
        model_arrays.update({
            f"{outcome_key}_feature_mean": outcome_model["feature_mean"],
            f"{outcome_key}_feature_scale": outcome_model["feature_scale"],
            f"{outcome_key}_outcome_coef": outcome_model["outcome_coef"],
        })
        predicted_values = expected_utility(probabilities, primary_lambda, eta)
        score = _advantage_score(predicted_values)
        rate_cal = calibrate_rate_threshold(
            score[calibration], data["sources"][calibration], target_rate,
        )
        threshold = float(rate_cal["threshold"])
        choice = _choice_from_values(predicted_values, threshold)
        methods.add(
            name,
            choice,
            predicted_values=predicted_values,
            outcome_probabilities_=probabilities,
            parameter_count=int(outcome_model["outcome_coef"].size),
            latency_us=_latency_us(
                lambda om=outcome_model, fx=features, th=threshold: _choice_from_values(
                    expected_utility(
                        predict_outcome_model(om, fx), primary_lambda, eta
                    ),
                    th,
                ), n
            ),
            metadata={
                "family": "outcome_router",
                "features": feature_name,
                "threshold": threshold,
                "deployable": True,
            },
        )
        fit_summaries[name] = outcome_fit
        calibration_records[name] = rate_cal

    # B13 diagnostic.  The tracked capture does not contain a deployable hazard
    # pose/distance feature.  Its condition tag is therefore explicitly oracle
    # metadata and is never allowed to trigger STOP-D by itself.
    geometry_choice = np.where(data["conditions"] == "glass", 1, 0)
    methods.add(
        "GeometryOracle-condition",
        geometry_choice,
        metadata={
            "family": "geometry",
            "deployable": False,
            "oracle_diagnostic_only": True,
            "rule": "Detour iff capture condition is glass",
        },
    )

    model_path = output_dir / "model_parameters.npz"
    np.savez_compressed(model_path, **model_arrays)

    predictions_path = output_dir / "all_predictions.npz"
    np.savez_compressed(
        predictions_path,
        method_names=np.asarray(methods.names),
        choices=np.stack(methods.choices),
        predicted_values=np.stack(methods.values),
        outcome_probabilities=np.stack(methods.outcome_probabilities),
        choice_probabilities=np.stack(methods.choice_probabilities),
        parameter_count=np.asarray(methods.parameter_counts, dtype=np.int64),
        latency_us_per_decision=np.asarray(methods.latencies, dtype=np.float64),
        decision_id=np.asarray([str(row["decision_id"]) for row in data["metadata"]]),
        source=np.asarray(data["sources"], dtype=str),
        split=np.asarray(data["splits"], dtype=str),
        condition=np.asarray(data["conditions"], dtype=str),
        horizon=np.asarray(data["horizons"], dtype=np.int64),
        outcomes=np.asarray(data["outcomes"], dtype=str),
    )

    config_snapshot = output_dir / "baseline_suite.yaml"
    config_snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    capture_files = (
        "capture_manifest.json",
        "decision_metadata.json",
        "option_rollouts.jsonl",
        "decision_features.npz",
    )
    primary_methods = [
        "Base",
        "Always Detour",
        "Always Retreat",
        "Risk->Retreat",
        "Risk->Detour",
        "Risk->BestFixed",
        "Risk+TwoStage",
        "DirectChoice",
        "DirectQ",
        "PairwiseAdvantage",
        "OutcomeRouter",
        "OutcomeRouter-hidden-only",
        "OutcomeRouter-state-action-only",
        "GeometryOracle-condition",
    ]
    manifest = {
        "schema_version": 1,
        "kind": "iclr27_router_baseline_suite",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "capture_root": str(capture),
        "capture_sha256": {name: _sha256(capture / name) for name in capture_files},
        "config_snapshot": config_snapshot.name,
        "config_sha256": _sha256(config_snapshot),
        "predictions": predictions_path.name,
        "model_parameters": model_path.name,
        "protocol": {
            "data_role": "development-only baseline audit; no fresh test outcomes",
            "primary_method": str(protocol["primary_method"]),
            "options": list(OPTIONS),
            "outcomes": list(OUTCOMES),
            "primary_lambda": primary_lambda,
            "noncompletion_cost": eta,
            "primary_target_intervention_rate": target_rate,
            "lambda_grid": lambda_grid,
            "option_costs": option_costs,
            "tie_break": "Base, then lower intervention cost, then option order",
            "feature_contract": {
                "primary": "single-frame train-PCA hidden + robot state + nominal action",
                "pca_components": int(pca[1].shape[1]),
                "pca_fit_split": "train",
                "source_weighting": "equal total weight per source",
            },
            "latency_scope": (
                "batched CPU projected-feature-to-choice head inference; excludes "
                "the frozen VLA forward pass and PCA projection"
            ),
            "go_thresholds": dict(protocol["go_thresholds"]),
            "go_comparators": dict(protocol["go_comparators"]),
            "fresh_test_prohibition": protocol.get("fresh_test_prohibition", ""),
            "primary_methods": primary_methods,
        },
        "source_counts": {
            split: len(set(data["sources"][data["splits"] == split]))
            for split in ("train", "calibration", "development")
        },
        "decision_counts": dict(Counter(map(str, data["splits"]))),
        "statistics": dict(config["statistics"]),
        "calibration": calibration_records,
        "fit_summaries": fit_summaries,
        "multi_lambda_direct_q": direct_q_models,
        "method_metadata": methods.metadata,
        "method_order": methods.names,
        "geometry_availability": {
            "deployable_geometry_features_in_capture": False,
            "reason": "capture has no current hazard pose/distance/clearance feature",
            "oracle_geometry_method": "GeometryOracle-condition",
            "stop_d_eligible": ["OutcomeRouter-state-action-only"],
        },
        "artifact_sha256": {
            predictions_path.name: _sha256(predictions_path),
            model_path.name: _sha256(model_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture",
        type=Path,
        default=ROOT / "results/counterfactual_router/full_d4751330395e_20260814T152231Z",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/iclr27/baseline_suite.yaml",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    manifest = train_suite(args.capture, config, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "git_commit": manifest["git_commit"],
        "methods": len(manifest["method_order"]),
        "source_counts": manifest["source_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
