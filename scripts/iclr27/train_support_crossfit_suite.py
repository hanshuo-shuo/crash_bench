#!/usr/bin/env python3
"""Train the Phase 2.5B source-cross-fitted support-rescue suite.

Every decision receives exactly one prediction from a model whose PCA,
standardization, fitting, hyperparameter selection, and rate calibration never
used that decision's source.  Historical split names are retained only for
provenance; all 20 sources form one exposed development corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.counterfactual_router import OPTIONS, OUTCOMES, outcome_probabilities
from scripts.iclr27.audit_option_support import strict_option_labels
from scripts.iclr27.train_router_baseline_suite import (
    _candidate_thresholds,
    _intervention_choice_cost_favoring,
    oracle_choice_base_favoring,
    realized_utility,
)
from scripts.train_minimal_counterfactual_router import (
    _source_weights,
    build_features,
    fit_frame_pca,
    load_capture,
)


STRICT_LABELS = ("strict_base", "strict_detour", "strict_retreat")
AUXILIARY_STRATUM = "auxiliary_tie_or_no_good"
LABEL_TO_CHOICE = {label: index for index, label in enumerate(STRICT_LABELS)}


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
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_native(rows))


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("support cross-fit config must be a mapping")
    return payload


def _hash_key(seed: int, outer_source: str, source: str, role: str = "outer") -> str:
    payload = f"{seed}|{role}|{outer_source}|{source}".encode()
    return hashlib.sha256(payload).hexdigest()


def outer_source_split(
    sources: Sequence[str], held_source: str, *, fit_sources: int, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministically divide the 19 non-held sources into 14 fit and 5 cal."""

    unique = sorted(set(map(str, sources)))
    if held_source not in unique:
        raise ValueError(f"unknown held source {held_source}")
    candidates = [source for source in unique if source != held_source]
    if len(candidates) != fit_sources + (len(unique) - 1 - fit_sources):
        raise AssertionError("source split arithmetic failed")
    ordered = sorted(
        candidates, key=lambda source: (_hash_key(seed, held_source, source), source)
    )
    return tuple(ordered[:fit_sources]), tuple(ordered[fit_sources:])


def _inner_source_groups(
    fit_sources: Sequence[str], *, folds: int, seed: int, outer_source: str
) -> list[tuple[str, ...]]:
    ordered = sorted(
        map(str, fit_sources),
        key=lambda source: (
            _hash_key(seed, outer_source, source, role="inner"), source
        ),
    )
    return [tuple(ordered[offset::folds]) for offset in range(folds)]


def _preflight_inputs(
    capture: Path, support_audit: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Reject forbidden split roles before opening option outcomes."""

    for name in config["inputs"]["required_capture_files"]:
        if not (capture / str(name)).is_file():
            raise FileNotFoundError(capture / str(name))
    for name in config["inputs"]["required_support_files"]:
        if not (support_audit / str(name)).is_file():
            raise FileNotFoundError(support_audit / str(name))

    metadata = json.loads((capture / "decision_metadata.json").read_text())
    observed_splits = {str(row["split"]) for row in metadata}
    allowed = set(map(str, config["data_policy"]["allowed_splits"]))
    forbidden = observed_splits - allowed
    if forbidden:
        raise ValueError(
            "Phase 2.5B refuses outcome-bearing forbidden splits: "
            + ", ".join(sorted(forbidden))
        )
    allocation: dict[str, str] = {}
    for row in metadata:
        source = str(row["source_state_sha256"])
        split = str(row["split"])
        previous = allocation.setdefault(source, split)
        if previous != split:
            raise ValueError(f"source {source} leaks across {previous}/{split}")
    expected_sources = int(config["data_policy"]["expected_source_count"])
    if len(allocation) != expected_sources:
        raise ValueError(
            f"expected {expected_sources} pooled sources, found {len(allocation)}"
        )
    expected_split_counts = config["data_policy"][
        "expected_historical_split_source_counts"
    ]
    actual_split_counts = Counter(allocation.values())
    if {key: int(actual_split_counts[key]) for key in expected_split_counts} != {
        str(key): int(value) for key, value in expected_split_counts.items()
    }:
        raise ValueError("historical split source counts changed")

    gate = json.loads((support_audit / "gate_decision.json").read_text())
    required_gate = str(config["inputs"]["required_support_gate"])
    if str(gate.get("decision")) != required_gate:
        raise ValueError(
            f"Phase 2.5B requires {required_gate}, found {gate.get('decision')}"
        )
    support_manifest = json.loads((support_audit / "manifest.json").read_text())
    expected_hashes = support_manifest.get("input_sha256", {})
    hashes: dict[str, str] = {}
    for name in config["inputs"]["required_capture_files"]:
        key = f"capture/{name}"
        actual = _sha256(capture / str(name))
        hashes[key] = actual
        if expected_hashes.get(key) != actual:
            raise ValueError(f"capture artifact differs from Phase 2.5A: {name}")
    hashes["support/manifest.json"] = _sha256(support_audit / "manifest.json")
    hashes["support/gate_decision.json"] = _sha256(
        support_audit / "gate_decision.json"
    )
    return metadata, hashes


def _source_macro_mean(values: np.ndarray, sources: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.average(values, weights=_source_weights(np.asarray(sources))))


def support_strata(labels: Sequence[str]) -> np.ndarray:
    return np.asarray([
        str(label) if str(label) in STRICT_LABELS else AUXILIARY_STRATUM
        for label in labels
    ])


def support_balanced_weights(
    sources: Sequence[str],
    strata: Sequence[str],
    *,
    cap_multiple: float,
    tolerance: float,
) -> np.ndarray:
    """Maximum-entropy balancing with exact source/stratum margins."""

    sources = np.asarray(sources, dtype=str)
    strata = np.asarray(strata, dtype=str)
    if len(sources) != len(strata) or not len(sources):
        raise ValueError("support balancing requires aligned non-empty arrays")
    source_levels = sorted(set(sources.tolist()))
    stratum_levels = sorted(set(strata.tolist()))
    balanced_strata = [
        level for level in stratum_levels if level != AUXILIARY_STRATUM
    ]
    if not balanced_strata:
        raise ValueError("support balancing requires at least one strict stratum")
    n = len(sources)
    source_target = n / len(source_levels)
    weights = np.ones(n, dtype=np.float64)
    converged = False
    for _ in range(10000):
        for source in source_levels:
            mask = sources == source
            total = float(weights[mask].sum())
            if total <= 0.0:
                raise RuntimeError(f"support balancing lost source mass: {source}")
            weights[mask] *= source_target / total
        strict_masses = [
            float(weights[strata == stratum].sum()) for stratum in balanced_strata
        ]
        stratum_target = float(np.mean(strict_masses))
        for stratum in balanced_strata:
            mask = strata == stratum
            total = float(weights[mask].sum())
            if total <= 0.0:
                raise RuntimeError(f"support balancing lost stratum mass: {stratum}")
            weights[mask] *= stratum_target / total
        source_error = max(
            abs(weights[sources == level].sum() - source_target)
            for level in source_levels
        )
        stratum_error = max(
            abs(weights[strata == level].sum() - stratum_target)
            for level in balanced_strata
        )
        if max(source_error, stratum_error) <= tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError("support balancing did not converge")
    source_totals = [weights[sources == level].sum() for level in source_levels]
    strict_totals = [weights[strata == level].sum() for level in balanced_strata]
    strict_target = float(np.mean(strict_totals))
    error = max(
        max(abs(value - source_target) for value in source_totals),
        max(abs(value - strict_target) for value in strict_totals),
    )
    if error > tolerance:
        raise RuntimeError(f"support balance residual {error:g} exceeds tolerance")
    if weights.max() > float(cap_multiple) + tolerance:
        raise RuntimeError("decision-weight cap was violated")
    return weights


def two_stage_support_weights(
    sources: Sequence[str],
    strata: Sequence[str],
    *,
    cap_multiple: float,
) -> np.ndarray:
    """Product weighting for the structurally sparse beneficial-only subset.

    Some folds make exact equal-source and Detour/Retreat margins infeasible
    because several sources contain only Detour-beneficial decisions.  This
    baseline therefore multiplies source weights by an inverse strict-class
    factor, while auxiliary ties retain their source weight.  M1/M2 continue
    to use :func:`support_balanced_weights` on the complete fitting corpus.
    """

    sources = np.asarray(sources, dtype=str)
    strata = np.asarray(strata, dtype=str)
    weights = _source_weights(sources)
    strict_levels = [
        level for level in ("strict_detour", "strict_retreat")
        if np.any(strata == level)
    ]
    if len(strict_levels) != 2:
        raise ValueError("two-stage support weighting requires both strict interventions")
    masses = {
        level: float(weights[strata == level].sum()) for level in strict_levels
    }
    target = float(np.mean(list(masses.values())))
    for level in strict_levels:
        weights[strata == level] *= target / masses[level]
    weights *= len(weights) / weights.sum()
    if weights.max() > float(cap_multiple) + 1e-12:
        raise RuntimeError("two-stage support weight exceeds the frozen cap")
    return weights


def _standardize_weighted(
    x: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance) + 1e-6
    return (x - mean) / scale, mean, scale


def fit_multinomial_weighted(
    x: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    classes: int,
    l2: float,
    max_iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64)
    xs, mean, scale = _standardize_weighted(x, weights)
    design = np.column_stack((xs, np.ones(len(xs))))
    rows = np.arange(len(labels))
    normalizer = float(np.sum(weights))

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coef = flat.reshape(design.shape[1], classes)
        probabilities = outcome_probabilities(design @ coef)
        loss = -float(np.sum(weights * np.log(probabilities[rows, labels] + 1e-12)))
        loss /= normalizer
        loss += 0.5 * float(l2) * float(np.sum(coef ** 2))
        residual = probabilities.copy()
        residual[rows, labels] -= 1.0
        residual *= weights[:, None] / normalizer
        gradient = design.T @ residual + float(l2) * coef
        return loss, gradient.ravel()

    fitted = minimize(
        objective,
        np.zeros(design.shape[1] * classes),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iterations), "ftol": 1e-11},
    )
    if not fitted.success:
        raise RuntimeError(f"multinomial fit failed: {fitted.message}")
    return {
        "mean": mean,
        "scale": scale,
        "coef": fitted.x.reshape(design.shape[1], classes),
    }, {
        "converged": True,
        "iterations": int(fitted.nit),
        "objective": float(fitted.fun),
    }


def predict_multinomial(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    design = np.column_stack((standardized, np.ones(len(standardized))))
    return outcome_probabilities(design @ model["coef"])


def fit_ridge_weighted(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, *, alpha: float
) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    xs, mean, scale = _standardize_weighted(x, weights)
    design = np.column_stack((xs, np.ones(len(xs))))
    weighted = design * np.sqrt(weights)[:, None]
    targets = y * np.sqrt(weights)[:, None] if y.ndim == 2 else y * np.sqrt(weights)
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(weighted.T @ weighted + penalty, weighted.T @ targets)
    return {"mean": mean, "scale": scale, "coef": coef}


def predict_ridge(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    return np.column_stack((standardized, np.ones(len(standardized)))) @ model["coef"]


def _outcome_labels(outcomes: np.ndarray) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(OUTCOMES)}
    return np.asarray([
        [lookup[str(value)] for value in row] for row in outcomes
    ], dtype=np.int64)


def _pair_rows(utility: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    decisions: list[int] = []
    left: list[int] = []
    right: list[int] = []
    signs: list[float] = []
    for decision, row in enumerate(np.asarray(utility, dtype=np.float64)):
        for option_a in range(len(OPTIONS)):
            for option_b in range(option_a + 1, len(OPTIONS)):
                difference = float(row[option_a] - row[option_b])
                if abs(difference) <= 1e-12:
                    continue
                decisions.append(decision)
                left.append(option_a)
                right.append(option_b)
                signs.append(float(np.sign(difference)))
    return (
        np.asarray(decisions, dtype=np.int64),
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(signs, dtype=np.float64),
    )


def fit_outcome_weighted(
    x: np.ndarray,
    outcomes: np.ndarray,
    utility: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
    beta: float,
    tau: float,
    outcome_values: Sequence[float],
    max_iterations: int,
    initial_coef: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit outcome softmax heads with an optional unequal-pair ranking loss."""

    labels = _outcome_labels(outcomes)
    xs, mean, scale = _standardize_weighted(x, weights)
    design = np.column_stack((xs, np.ones(len(xs))))
    rows = np.arange(len(labels))
    option_rows = np.arange(len(OPTIONS))
    normalizer = float(np.sum(weights) * len(OPTIONS))
    outcome_values = np.asarray(outcome_values, dtype=np.float64)
    pair_decision, pair_left, pair_right, pair_sign = _pair_rows(utility)
    pair_counts = Counter(pair_decision.tolist())
    pair_weights = np.asarray([
        weights[index] / pair_counts[int(index)] for index in pair_decision
    ], dtype=np.float64)
    pair_normalizer = float(pair_weights.sum()) if len(pair_weights) else 1.0

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coef = flat.reshape(design.shape[1], len(OPTIONS), len(OUTCOMES))
        logits = np.einsum("nd,dok->nok", design, coef)
        probabilities = outcome_probabilities(logits)
        log_selected = np.log(
            probabilities[rows[:, None], option_rows[None, :], labels] + 1e-12
        )
        loss = -float(np.sum(weights[:, None] * log_selected)) / normalizer
        residual = probabilities.copy()
        for option in range(len(OPTIONS)):
            residual[rows, option, labels[:, option]] -= 1.0
        residual *= weights[:, None, None] / normalizer

        rank_loss = 0.0
        if beta > 0.0 and len(pair_decision):
            predicted_values = probabilities @ outcome_values
            signed_margin = (
                (predicted_values[pair_decision, pair_left]
                 - predicted_values[pair_decision, pair_right])
                * pair_sign / float(tau)
            )
            rank_loss = float(
                np.sum(pair_weights * np.logaddexp(0.0, -signed_margin))
                / pair_normalizer
            )
            slope = (
                -pair_sign / float(tau)
                / (1.0 + np.exp(np.clip(signed_margin, -60.0, 60.0)))
                * pair_weights / pair_normalizer
            )
            value_gradient = np.zeros((len(x), len(OPTIONS)), dtype=np.float64)
            np.add.at(value_gradient, (pair_decision, pair_left), slope)
            np.add.at(value_gradient, (pair_decision, pair_right), -slope)
            expected = predicted_values[..., None]
            residual += float(beta) * value_gradient[..., None] * probabilities * (
                outcome_values[None, None, :] - expected
            )
            loss += float(beta) * rank_loss

        loss += 0.5 * float(l2) * float(np.sum(coef ** 2))
        gradient = np.einsum("nd,nok->dok", design, residual) + float(l2) * coef
        return loss, gradient.ravel()

    shape = (design.shape[1], len(OPTIONS), len(OUTCOMES))
    initial = np.zeros(shape, dtype=np.float64)
    if initial_coef is not None:
        initial = np.asarray(initial_coef, dtype=np.float64)
        if initial.shape != shape:
            raise ValueError("initial outcome coefficient shape changed")
    fitted = minimize(
        objective,
        initial.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iterations), "ftol": 1e-11},
    )
    if not fitted.success:
        raise RuntimeError(f"outcome/ranking fit failed: {fitted.message}")
    return {
        "mean": mean,
        "scale": scale,
        "coef": fitted.x.reshape(shape),
    }, {
        "converged": True,
        "iterations": int(fitted.nit),
        "objective": float(fitted.fun),
        "beta": float(beta),
        "tau": float(tau),
        "ranking_pairs": int(len(pair_decision)),
    }


def predict_outcome(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    design = np.column_stack((standardized, np.ones(len(standardized))))
    return outcome_probabilities(np.einsum("nd,dok->nok", design, model["coef"]))


def _expected_utility(
    probabilities: np.ndarray, *, catastrophe_cost: float, eta: float
) -> np.ndarray:
    values = {
        "task_success": 1.0,
        "catastrophe": -float(catastrophe_cost),
        "safe_noncompletion": -float(eta),
    }
    vector = np.asarray([values[name] for name in OUTCOMES], dtype=np.float64)
    return np.asarray(probabilities, dtype=np.float64) @ vector


def _choice_inputs_from_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intervention = 1 + np.argmax(values[:, 1:], axis=1)
    score = values[np.arange(len(values)), intervention] - values[:, 0]
    return score, intervention


def constrained_calibration(
    scores: np.ndarray,
    intervention_choice: np.ndarray,
    utility: np.ndarray,
    outcomes: np.ndarray,
    sources: np.ndarray,
    *,
    maximum_rate: float,
) -> dict[str, Any]:
    """Maximize calibration utility under the shared source-macro rate cap."""

    scores = np.asarray(scores, dtype=np.float64)
    intervention_choice = np.asarray(intervention_choice, dtype=np.int64)
    sources = np.asarray(sources)
    candidates: list[tuple[tuple[float, float, float, float], dict[str, Any]]] = []
    for threshold in _candidate_thresholds(scores):
        choice = np.where(scores > threshold, intervention_choice, 0)
        selected_utility = utility[np.arange(len(choice)), choice]
        selected_outcome = outcomes[np.arange(len(choice)), choice]
        rate = _source_macro_mean(choice != 0, sources)
        if rate > float(maximum_rate) + 1e-12:
            continue
        source_utility = _source_macro_mean(selected_utility, sources)
        catastrophe = _source_macro_mean(selected_outcome == "catastrophe", sources)
        record = {
            "threshold": float(threshold),
            "source_macro_utility": source_utility,
            "source_macro_catastrophe_rate": catastrophe,
            "source_macro_intervention_rate": rate,
            "maximum_intervention_rate": float(maximum_rate),
        }
        key = (source_utility, -catastrophe, -rate, float(threshold))
        candidates.append((key, record))
    if not candidates:
        raise RuntimeError("no calibration threshold satisfies the intervention cap")
    return max(candidates, key=lambda item: item[0])[1]


def _apply_threshold(
    scores: np.ndarray, intervention_choice: np.ndarray, threshold: float
) -> np.ndarray:
    return np.where(
        np.asarray(scores) > float(threshold), np.asarray(intervention_choice), 0
    ).astype(np.int64)


def _calibrate_best_fixed(
    risk_scores: np.ndarray,
    utility: np.ndarray,
    outcomes: np.ndarray,
    sources: np.ndarray,
    *,
    maximum_rate: float,
) -> dict[str, Any]:
    candidates = []
    for option in (1, 2):
        record = constrained_calibration(
            risk_scores,
            np.full(len(risk_scores), option),
            utility,
            outcomes,
            sources,
            maximum_rate=maximum_rate,
        )
        record = {**record, "fixed_option_index": option, "fixed_option": OPTIONS[option]}
        key = (
            record["source_macro_utility"],
            -record["source_macro_catastrophe_rate"],
            -record["source_macro_intervention_rate"],
            record["threshold"],
            -option,
        )
        candidates.append((key, record))
    return max(candidates, key=lambda item: item[0])[1]


def _weighted_macro_f1(
    choices: np.ndarray, labels: np.ndarray, sources: np.ndarray
) -> float:
    mask = np.isin(labels, STRICT_LABELS)
    if not np.any(mask):
        return float("nan")
    target = np.asarray([LABEL_TO_CHOICE[str(label)] for label in labels[mask]])
    predicted = np.asarray(choices)[mask]
    weights = _source_weights(np.asarray(sources)[mask])
    scores = []
    for option in range(len(OPTIONS)):
        tp = float(weights[(target == option) & (predicted == option)].sum())
        fp = float(weights[(target != option) & (predicted == option)].sum())
        fn = float(weights[(target == option) & (predicted != option)].sum())
        denominator = 2.0 * tp + fp + fn
        scores.append(0.0 if denominator <= 0.0 else 2.0 * tp / denominator)
    return float(np.mean(scores))


def _feature_views(
    arrays: Mapping[str, np.ndarray], pca: tuple[np.ndarray, np.ndarray]
) -> dict[str, np.ndarray]:
    full = build_features(arrays, pca, history=False, include_robot_action=True)
    hidden = np.asarray(arrays["hidden"], dtype=np.float64)[:, -1]
    hidden_only = (hidden - pca[0]) @ pca[1]
    state_action = np.concatenate((
        np.asarray(arrays["robot_state"], dtype=np.float64)[:, -1],
        np.asarray(arrays["nominal_action"], dtype=np.float64)[:, -1],
    ), axis=1)
    return {"full": full, "hidden": hidden_only, "state_action": state_action}


def _categorical_values(
    categories: np.ndarray,
    fit: np.ndarray,
    utility: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    categories = np.asarray(categories, dtype=str)
    values = np.empty_like(utility, dtype=np.float64)
    global_values = np.asarray([
        _source_macro_mean(utility[fit, option], sources[fit])
        for option in range(len(OPTIONS))
    ])
    for category in sorted(set(categories.tolist())):
        group_fit = fit & (categories == category)
        group_values = global_values
        if np.any(group_fit):
            group_values = np.asarray([
                _source_macro_mean(utility[group_fit, option], sources[group_fit])
                for option in range(len(OPTIONS))
            ])
        values[categories == category] = group_values
    return values


def _support_counts(
    sources: np.ndarray,
    labels: np.ndarray,
    margins: np.ndarray,
    fit: np.ndarray,
    *,
    gamma: float,
) -> dict[str, int]:
    return {
        label: len(set(sources[fit & (labels == label) & (margins >= gamma)].tolist()))
        for label in STRICT_LABELS
    }


def _model_arrays(prefix: str, model: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}_{key}": np.asarray(value) for key, value in model.items()}


def select_ranking_beta(
    *,
    data: Mapping[str, Any],
    outer_source: str,
    fit_source_names: Sequence[str],
    labels: np.ndarray,
    utility: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Select beta using only inner source folds contained in outer-fit sources."""

    model_config = config["model"]
    ranking = model_config["ranking"]
    betas = list(map(float, ranking["beta_grid"]))
    inner_groups = _inner_source_groups(
        fit_source_names,
        folds=int(ranking["inner_source_folds"]),
        seed=int(model_config["seed"]),
        outer_source=outer_source,
    )
    outcomes = np.asarray(data["outcomes"])
    sources = np.asarray(data["sources"], dtype=str)
    cap = float(model_config["support_balance"]["maximum_decision_weight_multiple"])
    tolerance = float(model_config["support_balance"]["equality_tolerance"])
    l2 = float(model_config["logistic_l2"])
    tau = float(ranking["tau"])
    max_iterations = int(model_config["optimizer_max_iterations"])
    rho = float(config["calibration"]["maximum_intervention_rate"])
    pca_components = int(config["features"]["pca_components"])
    outcome_value_map = config["utility"]["outcome_values"]
    outcome_value_vector = [float(outcome_value_map[name]) for name in OUTCOMES]
    details: dict[str, list[dict[str, Any]]] = {f"{beta:g}": [] for beta in betas}

    for inner_index, held_group in enumerate(inner_groups):
        inner_validation = np.isin(sources, held_group)
        inner_fit = np.isin(sources, fit_source_names) & ~inner_validation
        inner_counts = _support_counts(
            sources,
            labels,
            np.where(np.isin(labels, STRICT_LABELS), 1.0, 0.0),
            inner_fit,
            gamma=0.5,
        )
        if min(inner_counts.values()) < 1:
            for beta in betas:
                details[f"{beta:g}"].append({
                    "inner_fold": inner_index,
                    "validation_sources": list(held_group),
                    "support_valid": False,
                    "fit_source_support": inner_counts,
                })
            continue
        pca = fit_frame_pca(
            data["arrays"]["hidden"],
            data["arrays"]["history_mask"],
            inner_fit,
            pca_components,
            seed=int(model_config["seed"]) + inner_index,
        )
        x = _feature_views(data["arrays"], pca)["full"]
        inner_weights = support_balanced_weights(
            sources[inner_fit],
            support_strata(labels[inner_fit]),
            cap_multiple=cap,
            tolerance=tolerance,
        )
        base_model, _ = fit_outcome_weighted(
            x[inner_fit], outcomes[inner_fit], utility[inner_fit], inner_weights,
            l2=l2, beta=0.0, tau=tau, outcome_values=outcome_value_vector,
            max_iterations=max_iterations,
        )
        for beta in betas:
            model, summary = fit_outcome_weighted(
                x[inner_fit], outcomes[inner_fit], utility[inner_fit], inner_weights,
                l2=l2, beta=beta, tau=tau, outcome_values=outcome_value_vector,
                max_iterations=max_iterations, initial_coef=base_model["coef"],
            )
            probabilities = predict_outcome(model, x)
            values = _expected_utility(
                probabilities,
                catastrophe_cost=float(config["utility"]["lambda"]),
                eta=float(config["utility"]["eta"]),
            )
            scores, interventions = _choice_inputs_from_values(values)
            calibration = constrained_calibration(
                scores[inner_fit], interventions[inner_fit], utility[inner_fit],
                outcomes[inner_fit], sources[inner_fit], maximum_rate=rho,
            )
            choice = _apply_threshold(scores, interventions, calibration["threshold"])
            selected = utility[inner_validation, choice[inner_validation]]
            validation_utility = _source_macro_mean(
                selected, sources[inner_validation]
            )
            validation_f1 = _weighted_macro_f1(
                choice[inner_validation], labels[inner_validation], sources[inner_validation]
            )
            details[f"{beta:g}"].append({
                "inner_fold": inner_index,
                "validation_sources": list(held_group),
                "support_valid": True,
                "fit_source_support": inner_counts,
                "validation_source_macro_utility": validation_utility,
                "validation_strict_choice_macro_f1": validation_f1,
                "fit_summary": summary,
            })

    candidates = []
    summaries = {}
    for beta in betas:
        valid = [row for row in details[f"{beta:g}"] if row["support_valid"]]
        if not valid:
            raise RuntimeError(f"no support-valid inner folds for beta={beta:g}")
        utility_mean = float(np.mean([
            row["validation_source_macro_utility"] for row in valid
        ]))
        f1_mean = float(np.nanmean([
            row["validation_strict_choice_macro_f1"] for row in valid
        ]))
        summaries[f"{beta:g}"] = {
            "valid_inner_folds": len(valid),
            "mean_source_macro_utility": utility_mean,
            "mean_strict_choice_macro_f1": f1_mean,
        }
        candidates.append(((utility_mean, f1_mean, -beta), beta))
    selected = max(candidates, key=lambda row: row[0])[1]
    return {
        "outer_source": outer_source,
        "selected_beta": selected,
        "candidate_summary": summaries,
        "inner_folds": details,
    }


class OOFPredictions:
    def __init__(self, methods: Sequence[str], decisions: int) -> None:
        self.methods = list(methods)
        self.index = {name: offset for offset, name in enumerate(self.methods)}
        self.choice = np.full((len(methods), decisions), -1, dtype=np.int16)
        self.values = np.full((len(methods), decisions, len(OPTIONS)), np.nan)
        self.probabilities = np.full(
            (len(methods), decisions, len(OPTIONS), len(OUTCOMES)), np.nan
        )

    def set(
        self,
        method: str,
        held: np.ndarray,
        choice: np.ndarray,
        *,
        values: np.ndarray | None = None,
        probabilities: np.ndarray | None = None,
    ) -> None:
        index = self.index[method]
        self.choice[index, held] = np.asarray(choice, dtype=np.int16)[held]
        if values is not None:
            self.values[index, held] = np.asarray(values)[held]
        if probabilities is not None:
            self.probabilities[index, held] = np.asarray(probabilities)[held]

    def validate_complete(self) -> None:
        missing = {
            method: int(np.sum(self.choice[index] < 0))
            for index, method in enumerate(self.methods)
            if np.any(self.choice[index] < 0)
        }
        if missing:
            raise RuntimeError(f"missing OOF choices: {missing}")


def train_crossfit_suite(
    *,
    capture: Path,
    support_audit: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)
    capture = capture.resolve()
    support_audit = support_audit.resolve()
    output_dir = output_dir.resolve()
    metadata, input_hashes = _preflight_inputs(capture, support_audit, config)
    output_dir.mkdir(parents=True, exist_ok=False)

    catastrophe_cost = float(config["utility"]["lambda"])
    eta = float(config["utility"]["eta"])
    data = load_capture(capture, catastrophe_cost)
    if len(metadata) != len(data["metadata"]):
        raise ValueError("metadata changed during capture load")
    sources = np.asarray(data["sources"], dtype=str)
    outcomes = np.asarray(data["outcomes"], dtype=str)
    utility = realized_utility(outcomes, catastrophe_cost, eta)
    labels, margins = strict_option_labels(utility, catastrophe_utility=-catastrophe_cost)
    unique_sources = sorted(set(sources.tolist()))

    method_order = (
        list(config["methods"]["fixed"])
        + list(config["methods"]["scalar_risk"])
        + list(config["methods"]["learned"])
        + list(config["methods"]["diagnostic_only"])
    )
    predictions = OOFPredictions(method_order, len(sources))
    outer = config["data_policy"]["outer_crossfit"]
    fit_source_count = int(outer["fit_sources"])
    split_seed = int(outer["seed"])
    gamma = float(config["data_policy"]["support_validity"]["support_advantage_gamma"])
    minimum_support = int(
        config["data_policy"]["support_validity"]["minimum_fit_sources_per_label"]
    )
    rho = float(config["calibration"]["maximum_intervention_rate"])
    model_config = config["model"]
    l2 = float(model_config["logistic_l2"])
    ridge_alpha = float(model_config["ridge_alpha"])
    max_iterations = int(model_config["optimizer_max_iterations"])
    pca_components = int(config["features"]["pca_components"])
    cap = float(model_config["support_balance"]["maximum_decision_weight_multiple"])
    tolerance = float(model_config["support_balance"]["equality_tolerance"])
    tau = float(model_config["ranking"]["tau"])
    option_costs = list(map(float, config["utility"]["option_costs"]))
    outcome_value_map = config["utility"]["outcome_values"]
    outcome_value_vector = [float(outcome_value_map[name]) for name in OUTCOMES]

    fold_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    calibration_records: dict[str, Any] = {}
    fit_summaries: dict[str, Any] = {}
    beta_records: dict[str, Any] = {}
    model_arrays: dict[str, np.ndarray] = {}

    for fold_index, held_source in enumerate(unique_sources):
        fit_names, calibration_names = outer_source_split(
            unique_sources, held_source, fit_sources=fit_source_count, seed=split_seed
        )
        held = sources == held_source
        fit = np.isin(sources, fit_names)
        calibration = np.isin(sources, calibration_names)
        if np.any(held & (fit | calibration)) or np.any(fit & calibration):
            raise AssertionError("outer source roles overlap")
        if not np.all(held | fit | calibration):
            raise AssertionError("outer source roles do not cover the corpus")
        counts = _support_counts(
            sources, labels, margins, fit, gamma=gamma
        )
        support_valid = min(counts.values()) >= minimum_support
        for label in STRICT_LABELS:
            support_rows.append({
                "outer_fold": fold_index,
                "held_source": held_source,
                "strict_label": label,
                "fit_source_support": counts[label],
                "minimum_required": minimum_support,
                "support_valid": support_valid,
            })
        for source in unique_sources:
            role = "held_out" if source == held_source else (
                "fit" if source in fit_names else "calibration"
            )
            fold_rows.append({
                "outer_fold": fold_index,
                "held_source": held_source,
                "source": source,
                "role": role,
                "assignment_hash": (
                    "" if role == "held_out" else _hash_key(split_seed, held_source, source)
                ),
            })
        if not support_valid:
            _write_csv(output_dir / "fold_assignments.csv", fold_rows)
            _write_csv(output_dir / "fold_support.csv", support_rows)
            raise RuntimeError(
                f"outer fold {fold_index} held={held_source} is support_invalid: {counts}"
            )

        pca = fit_frame_pca(
            data["arrays"]["hidden"], data["arrays"]["history_mask"], fit,
            pca_components, seed=int(model_config["seed"]) + fold_index,
        )
        features = _feature_views(data["arrays"], pca)
        x_full = features["full"]
        source_weights = _source_weights(sources[fit])
        balanced_weights = support_balanced_weights(
            sources[fit], support_strata(labels[fit]), cap_multiple=cap,
            tolerance=tolerance,
        )
        prefix = f"fold_{fold_index:02d}"
        model_arrays[f"{prefix}_pca_mean"] = pca[0]
        model_arrays[f"{prefix}_pca_components"] = pca[1]

        predictions.set("Base", held, np.zeros(len(sources), dtype=np.int64))
        predictions.set("Always Detour", held, np.ones(len(sources), dtype=np.int64))
        predictions.set(
            "Always Retreat", held, np.full(len(sources), 2, dtype=np.int64)
        )

        base_catastrophe = (outcomes[:, 0] == "catastrophe").astype(np.int64)
        risk_model, risk_summary = fit_multinomial_weighted(
            x_full[fit], base_catastrophe[fit], source_weights,
            classes=2, l2=l2, max_iterations=max_iterations,
        )
        risk_probability = predict_multinomial(risk_model, x_full)[:, 1]
        model_arrays.update(_model_arrays(f"{prefix}_risk", risk_model))
        fit_summaries[f"{prefix}/risk"] = risk_summary
        for method, option in (("Risk->Retreat", 2), ("Risk->Detour", 1)):
            record = constrained_calibration(
                risk_probability[calibration],
                np.full(int(calibration.sum()), option),
                utility[calibration], outcomes[calibration], sources[calibration],
                maximum_rate=rho,
            )
            choice = _apply_threshold(
                risk_probability, np.full(len(sources), option), record["threshold"]
            )
            predictions.set(method, held, choice)
            calibration_records[f"{prefix}/{method}"] = record
        best_fixed = _calibrate_best_fixed(
            risk_probability[calibration], utility[calibration], outcomes[calibration],
            sources[calibration], maximum_rate=rho,
        )
        fixed_option = int(best_fixed["fixed_option_index"])
        fixed_choice = _apply_threshold(
            risk_probability, np.full(len(sources), fixed_option), best_fixed["threshold"]
        )
        predictions.set("Risk->BestFixed", held, fixed_choice)
        calibration_records[f"{prefix}/Risk->BestFixed"] = best_fixed

        intervention_utility = utility[:, 1:]
        beneficial = intervention_utility.max(axis=1) > utility[:, 0] + 1e-12
        intervention_target = 1 + _intervention_choice_cost_favoring(
            intervention_utility, option_costs[1:]
        )
        option_fit = fit & beneficial
        ordinary_option_weights = _source_weights(sources[option_fit])
        option_strata = np.asarray([
            label if label in ("strict_detour", "strict_retreat") else AUXILIARY_STRATUM
            for label in labels[option_fit]
        ])
        sb_option_weights = two_stage_support_weights(
            sources[option_fit], option_strata, cap_multiple=cap
        )
        for method, weights in (
            ("Risk+TwoStage", ordinary_option_weights),
            ("SB-Risk+TwoStage", sb_option_weights),
        ):
            option_model, option_summary = fit_multinomial_weighted(
                x_full[option_fit], intervention_target[option_fit] - 1, weights,
                classes=2, l2=l2, max_iterations=max_iterations,
            )
            option_probability = predict_multinomial(option_model, x_full)
            intervention_choice = 1 + np.argmax(option_probability, axis=1)
            record = constrained_calibration(
                risk_probability[calibration], intervention_choice[calibration],
                utility[calibration], outcomes[calibration], sources[calibration],
                maximum_rate=rho,
            )
            choice = _apply_threshold(
                risk_probability, intervention_choice, record["threshold"]
            )
            predictions.set(method, held, choice)
            key = method.lower().replace("+", "_").replace("-", "_")
            model_arrays.update(_model_arrays(f"{prefix}_{key}", option_model))
            fit_summaries[f"{prefix}/{method}"] = option_summary
            calibration_records[f"{prefix}/{method}"] = record

        direct_labels = oracle_choice_base_favoring(utility, option_costs)
        direct_model, direct_summary = fit_multinomial_weighted(
            x_full[fit], direct_labels[fit], source_weights,
            classes=3, l2=l2, max_iterations=max_iterations,
        )
        direct_probability = predict_multinomial(direct_model, x_full)
        direct_intervention = 1 + np.argmax(direct_probability[:, 1:], axis=1)
        direct_score = (
            direct_probability[np.arange(len(sources)), direct_intervention]
            - direct_probability[:, 0]
        )
        direct_cal = constrained_calibration(
            direct_score[calibration], direct_intervention[calibration],
            utility[calibration], outcomes[calibration], sources[calibration],
            maximum_rate=rho,
        )
        direct_choice = _apply_threshold(
            direct_score, direct_intervention, direct_cal["threshold"]
        )
        predictions.set("DirectChoice", held, direct_choice)
        model_arrays.update(_model_arrays(f"{prefix}_direct_choice", direct_model))
        fit_summaries[f"{prefix}/DirectChoice"] = direct_summary
        calibration_records[f"{prefix}/DirectChoice"] = direct_cal

        for method, weights in (
            ("DirectQ", source_weights), ("SB-DirectQ", balanced_weights)
        ):
            q_model = fit_ridge_weighted(
                x_full[fit], utility[fit], weights, alpha=ridge_alpha
            )
            q_values = predict_ridge(q_model, x_full)
            q_score, q_intervention = _choice_inputs_from_values(q_values)
            q_cal = constrained_calibration(
                q_score[calibration], q_intervention[calibration], utility[calibration],
                outcomes[calibration], sources[calibration], maximum_rate=rho,
            )
            q_choice = _apply_threshold(q_score, q_intervention, q_cal["threshold"])
            predictions.set(method, held, q_choice, values=q_values)
            key = method.lower().replace("-", "_")
            model_arrays.update(_model_arrays(f"{prefix}_{key}", q_model))
            calibration_records[f"{prefix}/{method}"] = q_cal

        advantage_target = utility[:, 1:] - utility[:, [0]]
        advantage_model = fit_ridge_weighted(
            x_full[fit], advantage_target[fit], source_weights, alpha=ridge_alpha
        )
        predicted_advantage = predict_ridge(advantage_model, x_full)
        advantage_values = np.column_stack((np.zeros(len(sources)), predicted_advantage))
        advantage_score, advantage_intervention = _choice_inputs_from_values(
            advantage_values
        )
        advantage_cal = constrained_calibration(
            advantage_score[calibration], advantage_intervention[calibration],
            utility[calibration], outcomes[calibration], sources[calibration],
            maximum_rate=rho,
        )
        advantage_choice = _apply_threshold(
            advantage_score, advantage_intervention, advantage_cal["threshold"]
        )
        predictions.set(
            "PairwiseAdvantage", held, advantage_choice, values=advantage_values
        )
        model_arrays.update(_model_arrays(f"{prefix}_pairwise", advantage_model))
        calibration_records[f"{prefix}/PairwiseAdvantage"] = advantage_cal

        outcome_variants = (
            ("OutcomeRouter", features["full"], source_weights),
            ("SB-OutcomeRouter", features["full"], balanced_weights),
            ("OutcomeRouter-hidden-only", features["hidden"], source_weights),
            ("OutcomeRouter-state-action-only", features["state_action"], source_weights),
        )
        sb_outcome_model: dict[str, np.ndarray] | None = None
        for method, feature_matrix, weights in outcome_variants:
            outcome_model, outcome_summary = fit_outcome_weighted(
                feature_matrix[fit], outcomes[fit], utility[fit], weights,
                l2=l2, beta=0.0, tau=tau, outcome_values=outcome_value_vector,
                max_iterations=max_iterations,
            )
            probabilities = predict_outcome(outcome_model, feature_matrix)
            values = _expected_utility(
                probabilities, catastrophe_cost=catastrophe_cost, eta=eta
            )
            score, intervention = _choice_inputs_from_values(values)
            record = constrained_calibration(
                score[calibration], intervention[calibration], utility[calibration],
                outcomes[calibration], sources[calibration], maximum_rate=rho,
            )
            choice = _apply_threshold(score, intervention, record["threshold"])
            predictions.set(
                method, held, choice, values=values, probabilities=probabilities
            )
            key = method.lower().replace("-", "_")
            model_arrays.update(_model_arrays(f"{prefix}_{key}", outcome_model))
            fit_summaries[f"{prefix}/{method}"] = outcome_summary
            calibration_records[f"{prefix}/{method}"] = record
            if method == "SB-OutcomeRouter":
                sb_outcome_model = outcome_model

        beta_record = select_ranking_beta(
            data=data, outer_source=held_source, fit_source_names=fit_names,
            labels=labels, utility=utility, config=config,
        )
        beta_records[prefix] = beta_record
        if sb_outcome_model is None:
            raise AssertionError("support-balanced outcome initialization missing")
        sbcor_model, sbcor_summary = fit_outcome_weighted(
            x_full[fit], outcomes[fit], utility[fit], balanced_weights,
            l2=l2, beta=float(beta_record["selected_beta"]), tau=tau,
            outcome_values=outcome_value_vector, max_iterations=max_iterations,
            initial_coef=sb_outcome_model["coef"],
        )
        sbcor_probabilities = predict_outcome(sbcor_model, x_full)
        sbcor_values = _expected_utility(
            sbcor_probabilities, catastrophe_cost=catastrophe_cost, eta=eta
        )
        sbcor_score, sbcor_intervention = _choice_inputs_from_values(sbcor_values)
        sbcor_cal = constrained_calibration(
            sbcor_score[calibration], sbcor_intervention[calibration],
            utility[calibration], outcomes[calibration], sources[calibration],
            maximum_rate=rho,
        )
        sbcor_choice = _apply_threshold(
            sbcor_score, sbcor_intervention, sbcor_cal["threshold"]
        )
        predictions.set(
            "SB-COR", held, sbcor_choice, values=sbcor_values,
            probabilities=sbcor_probabilities,
        )
        model_arrays.update(_model_arrays(f"{prefix}_sb_cor", sbcor_model))
        fit_summaries[f"{prefix}/SB-COR"] = sbcor_summary
        calibration_records[f"{prefix}/SB-COR"] = sbcor_cal

        for method, categories in (
            ("Condition-only diagnostic", np.asarray(data["conditions"], dtype=str)),
            ("Horizon-only diagnostic", np.asarray(data["horizons"], dtype=str)),
        ):
            diagnostic_values = _categorical_values(
                categories, fit, utility, sources
            )
            score, intervention = _choice_inputs_from_values(diagnostic_values)
            record = constrained_calibration(
                score[calibration], intervention[calibration], utility[calibration],
                outcomes[calibration], sources[calibration], maximum_rate=rho,
            )
            choice = _apply_threshold(score, intervention, record["threshold"])
            predictions.set(method, held, choice, values=diagnostic_values)
            calibration_records[f"{prefix}/{method}"] = record

    predictions.validate_complete()
    _write_csv(output_dir / "fold_assignments.csv", fold_rows)
    _write_csv(output_dir / "fold_support.csv", support_rows)
    (output_dir / "calibration.json").write_text(
        json.dumps(_native(calibration_records), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "beta_selection.json").write_text(
        json.dumps(_native(beta_records), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "fit_summaries.json").write_text(
        json.dumps(_native(fit_summaries), indent=2, sort_keys=True) + "\n"
    )
    model_path = output_dir / "model_parameters.npz"
    np.savez_compressed(model_path, **model_arrays)
    prediction_path = output_dir / "all_predictions.npz"
    np.savez_compressed(
        prediction_path,
        method_names=np.asarray(method_order),
        choices=predictions.choice,
        predicted_values=predictions.values,
        outcome_probabilities=predictions.probabilities,
        decision_id=np.asarray([str(row["decision_id"]) for row in data["metadata"]]),
        source=sources,
        historical_split=np.asarray(data["splits"], dtype=str),
        condition=np.asarray(data["conditions"], dtype=str),
        horizon=np.asarray(data["horizons"], dtype=np.int64),
        outcomes=outcomes,
        strict_label=np.asarray(labels, dtype=str),
        strict_advantage=np.asarray(margins, dtype=np.float64),
    )
    config_snapshot = output_dir / "support_crossfit.yaml"
    config_snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    artifact_hashes = {
        path.name: _sha256(path)
        for path in (
            output_dir / "fold_assignments.csv",
            output_dir / "fold_support.csv",
            output_dir / "calibration.json",
            output_dir / "beta_selection.json",
            output_dir / "fit_summaries.json",
            model_path,
            prediction_path,
            config_snapshot,
        )
    }
    manifest = {
        "schema_version": 1,
        "kind": "iclr27_support_crossfit_suite",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_dirty(),
        "evidence_level": "exposed-development-only",
        "new_rollouts": False,
        "fresh_test_outcomes_loaded": False,
        "capture_root": str(capture),
        "support_audit_root": str(support_audit),
        "input_sha256": input_hashes,
        "config_source": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "config_snapshot": config_snapshot.name,
        "source_count": len(unique_sources),
        "decision_count": len(sources),
        "outer_folds": len(unique_sources),
        "outer_fit_sources": fit_source_count,
        "outer_calibration_sources": len(unique_sources) - fit_source_count - 1,
        "all_outer_folds_support_valid": all(row["support_valid"] for row in support_rows),
        "method_order": method_order,
        "method_roles": {
            **{name: "fixed" for name in config["methods"]["fixed"]},
            **{name: "scalar_risk" for name in config["methods"]["scalar_risk"]},
            **{name: "learned" for name in config["methods"]["learned"]},
            **{name: "diagnostic_only" for name in config["methods"]["diagnostic_only"]},
        },
        "protocol": {
            "outer_crossfit": dict(outer),
            "pca_scale_fit_role": "outer fit sources only",
            "threshold_selection_role": "outer calibration sources only",
            "ranking_beta_selection_role": "inner folds of outer fit sources only",
            "maximum_source_macro_intervention_rate": rho,
            "primary_lambda": catastrophe_cost,
            "eta": eta,
            "strict_ties_used_as_support": False,
        },
        "statistics": dict(config["statistics"]),
        "artifact_sha256": artifact_hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_native(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/iclr27/support_crossfit.yaml",
    )
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--support-audit", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    capture = (
        args.capture.resolve() if args.capture
        else (ROOT / config["inputs"]["capture_path"]).resolve()
    )
    support_audit = (
        args.support_audit.resolve() if args.support_audit
        else (ROOT / config["inputs"]["option_support_audit_path"]).resolve()
    )
    manifest = train_crossfit_suite(
        capture=capture,
        support_audit=support_audit,
        config_path=args.config.resolve(),
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "git_commit": manifest["git_commit"],
        "outer_folds": manifest["outer_folds"],
        "methods": len(manifest["method_order"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
