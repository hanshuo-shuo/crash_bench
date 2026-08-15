#!/usr/bin/env python3
"""Train and evaluate a minimal source-disjoint counterfactual option router.

The model is intentionally linear: a train-only shared frame PCA is followed by
source-balanced ridge regression onto the realized utility of each option.  A
single Base-favoring intervention margin is selected on calibration sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from crashbench.counterfactual_router import OPTIONS, OUTCOMES


DEFAULT_UTILITY = {
    "task_success": 1.0,
    "safe_noncompletion": 0.0,
    "catastrophe": -5.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_weights(sources: np.ndarray) -> np.ndarray:
    counts = Counter(str(source) for source in sources)
    weights = np.asarray([1.0 / counts[str(source)] for source in sources])
    return weights * (len(weights) / weights.sum())


def fit_frame_pca(
    hidden: np.ndarray,
    history_mask: np.ndarray,
    train: np.ndarray,
    components: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic randomized PCA on valid training-history frames."""

    frames = np.asarray(hidden[train], dtype=np.float64)[history_mask[train] > 0]
    if len(frames) < 2:
        raise ValueError("PCA requires at least two valid training frames")
    mean = frames.mean(axis=0)
    centered = frames - mean
    k = min(int(components), *centered.shape)
    if k < 1:
        raise ValueError("PCA component count must be positive")
    rank = min(k + 8, *centered.shape)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((centered.shape[1], rank))
    q, _ = np.linalg.qr(centered @ omega, mode="reduced")
    for _ in range(2):
        q, _ = np.linalg.qr(centered.T @ q, mode="reduced")
        q, _ = np.linalg.qr(centered @ q, mode="reduced")
    _, _, vt = np.linalg.svd(q.T @ centered, full_matrices=False)
    return mean, vt[:k].T


def build_features(
    arrays: Mapping[str, np.ndarray],
    pca: tuple[np.ndarray, np.ndarray],
    *,
    history: bool,
    include_robot_action: bool,
) -> np.ndarray:
    hidden = np.asarray(arrays["hidden"], dtype=np.float64)
    mask = np.asarray(arrays["history_mask"], dtype=np.float64)
    mean, components = pca
    projected = np.einsum("ntd,dk->ntk", hidden - mean, components)
    projected *= mask[..., None]
    if history:
        blocks = [projected.reshape(len(projected), -1), mask]
        if include_robot_action:
            robot = np.asarray(arrays["robot_state"], dtype=np.float64) * mask[..., None]
            action = np.asarray(arrays["nominal_action"], dtype=np.float64) * mask[..., None]
            blocks.extend((robot.reshape(len(robot), -1), action.reshape(len(action), -1)))
    else:
        blocks = [projected[:, -1]]
        if include_robot_action:
            blocks.extend((
                np.asarray(arrays["robot_state"], dtype=np.float64)[:, -1],
                np.asarray(arrays["nominal_action"], dtype=np.float64)[:, -1],
            ))
    return np.concatenate(blocks, axis=1)


def fit_weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Fit standardized multi-output ridge with equal total weight per source."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = _source_weights(np.asarray(sources))
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance) + 1e-6
    xs = (x - mean) / scale
    design = np.column_stack((xs, np.ones(len(xs))))
    weighted = design * np.sqrt(weights)[:, None]
    targets = y * np.sqrt(weights)[:, None] if y.ndim == 2 else y * np.sqrt(weights)
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(weighted.T @ weighted + penalty, weighted.T @ targets)
    return {"mean": mean, "scale": scale, "coef": coef}


def predict_ridge(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    return np.column_stack((xs, np.ones(len(xs)))) @ model["coef"]


def load_capture(root: Path, catastrophe_cost: float) -> dict[str, Any]:
    metadata = json.loads((root / "decision_metadata.json").read_text())
    rows = [
        json.loads(line)
        for line in (root / "option_rollouts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_decision: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        by_decision[str(row["decision_id"])][str(row["option"])] = str(row["outcome"])
    metadata = sorted(metadata, key=lambda row: int(row["feature_index"]))
    if [int(row["feature_index"]) for row in metadata] != list(range(len(metadata))):
        raise ValueError("decision feature indices are not contiguous")
    utility_map = dict(DEFAULT_UTILITY)
    utility_map["catastrophe"] = -float(catastrophe_cost)
    outcomes = np.empty((len(metadata), len(OPTIONS)), dtype=object)
    utility = np.empty((len(metadata), len(OPTIONS)), dtype=np.float64)
    for index, row in enumerate(metadata):
        option_outcomes = by_decision[str(row["decision_id"])]
        if set(option_outcomes) != set(OPTIONS):
            raise ValueError(f"incomplete decision {row['decision_id']}")
        for option_index, option in enumerate(OPTIONS):
            outcome = option_outcomes[option]
            outcomes[index, option_index] = outcome
            utility[index, option_index] = utility_map[outcome]
    with np.load(root / "decision_features.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(len(value) != len(metadata) for value in arrays.values()):
        raise ValueError("feature archive and decision metadata disagree")
    return {
        "metadata": metadata,
        "arrays": arrays,
        "outcomes": outcomes,
        "utility": utility,
        "utility_map": utility_map,
        "splits": np.asarray([row["split"] for row in metadata]),
        "sources": np.asarray([row["source_state_sha256"] for row in metadata]),
        "conditions": np.asarray([row["condition"] for row in metadata]),
        "horizons": np.asarray([int(row["horizon_actions"]) for row in metadata]),
    }


def choose_router(predicted: np.ndarray, margin: float) -> np.ndarray:
    intervention = 1 + np.argmax(predicted[:, 1:], axis=1)
    advantage = predicted[np.arange(len(predicted)), intervention] - predicted[:, 0]
    return np.where(advantage > margin, intervention, 0).astype(np.int64)


def _candidate_margins(predicted: np.ndarray) -> list[float]:
    intervention = 1 + np.argmax(predicted[:, 1:], axis=1)
    advantage = predicted[np.arange(len(predicted)), intervention] - predicted[:, 0]
    positive = sorted(set(float(value) for value in advantage if value >= 0.0))
    candidates = [0.0]
    candidates.extend((left + right) / 2 for left, right in zip(positive, positive[1:]))
    candidates.append(float("inf"))
    return sorted(set(candidates))


def calibrate_router_margin(
    predicted: np.ndarray, utility: np.ndarray, sources: np.ndarray
) -> tuple[float, dict[str, float]]:
    weights = _source_weights(sources)
    candidates = []
    for margin in _candidate_margins(predicted):
        choice = choose_router(predicted, margin)
        realized = utility[np.arange(len(choice)), choice]
        candidates.append((
            float(np.average(realized, weights=weights)),
            -float(np.mean(choice != 0)),
            -margin,
            margin,
        ))
    best = max(candidates)
    return float(best[-1]), {
        "source_macro_utility": best[0],
        "intervention_rate": -best[1],
    }


def calibrate_risk_gate(
    scores: np.ndarray, utility: np.ndarray, sources: np.ndarray
) -> tuple[float, int, dict[str, float]]:
    unique = sorted(set(float(value) for value in scores))
    thresholds = [float("inf"), float("-inf")]
    thresholds.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    weights = _source_weights(sources)
    candidates = []
    for option in (1, 2):
        for threshold in thresholds:
            choice = np.where(scores > threshold, option, 0)
            realized = utility[np.arange(len(choice)), choice]
            candidates.append((
                float(np.average(realized, weights=weights)),
                -float(np.mean(choice != 0)),
                -option,
                threshold,
                option,
            ))
    best = max(candidates)
    return float(best[-2]), int(best[-1]), {
        "source_macro_utility": best[0],
        "intervention_rate": -best[1],
    }


def evaluate_choices(
    choice: np.ndarray,
    outcomes: np.ndarray,
    utility: np.ndarray,
    sources: np.ndarray,
) -> dict[str, Any]:
    indices = np.arange(len(choice))
    realized_outcomes = outcomes[indices, choice]
    realized_utility = utility[indices, choice]
    oracle_utility = utility.max(axis=1)
    source_means = [
        float(realized_utility[sources == source].mean()) for source in sorted(set(sources))
    ]
    counts = Counter(str(value) for value in realized_outcomes)
    base_mean = float(utility[:, 0].mean())
    oracle_mean = float(oracle_utility.mean())
    mean_utility = float(realized_utility.mean())
    denominator = oracle_mean - base_mean
    return {
        "decisions": len(choice),
        "choice_counts": {option: int(np.sum(choice == i)) for i, option in enumerate(OPTIONS)},
        "outcomes": {outcome: counts.get(outcome, 0) for outcome in OUTCOMES},
        "rates": {outcome: counts.get(outcome, 0) / len(choice) for outcome in OUTCOMES},
        "intervention_rate": float(np.mean(choice != 0)),
        "mean_utility": mean_utility,
        "source_macro_mean_utility": float(np.mean(source_means)),
        "mean_oracle_regret": float(np.mean(oracle_utility - realized_utility)),
        "optimal_option_rate": float(np.mean(realized_utility == oracle_utility)),
        "decision_value_recovered": (
            (mean_utility - base_mean) / denominator if denominator > 0 else 0.0
        ),
    }


def evaluate_method_by_group(
    choice: np.ndarray,
    data: Mapping[str, Any],
    indices: np.ndarray,
    group_values: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for value in sorted(set(group_values[indices]), key=str):
        local = group_values[indices] == value
        selected = indices[local]
        result[str(value)] = evaluate_choices(
            choice[local], data["outcomes"][selected], data["utility"][selected],
            data["sources"][selected],
        )
    return result


def train_variant(
    data: Mapping[str, Any],
    *,
    history: bool,
    include_robot_action: bool,
    pca_components: int,
    ridge_alpha: float,
    train_horizon: int | None = None,
    evaluation_horizon: int | None = None,
) -> dict[str, Any]:
    train = data["splits"] == "train"
    calibration = data["splits"] == "calibration"
    development = data["splits"] == "development"
    if train_horizon is not None:
        train &= data["horizons"] == train_horizon
        calibration &= data["horizons"] == train_horizon
    if evaluation_horizon is not None:
        development &= data["horizons"] == evaluation_horizon
    pca = fit_frame_pca(
        data["arrays"]["hidden"], data["arrays"]["history_mask"], train,
        pca_components,
    )
    x = build_features(
        data["arrays"], pca, history=history,
        include_robot_action=include_robot_action,
    )
    model = fit_weighted_ridge(
        x[train], data["utility"][train], data["sources"][train], ridge_alpha,
    )
    predicted = predict_ridge(model, x)
    margin, calibration_summary = calibrate_router_margin(
        predicted[calibration], data["utility"][calibration],
        data["sources"][calibration],
    )
    dev_indices = np.flatnonzero(development)
    raw_choice = choose_router(predicted[development], 0.0)
    calibrated_choice = choose_router(predicted[development], margin)
    report = {
        "feature_contract": {
            "history": "eight_frame" if history else "single_frame",
            "inputs": "hidden_robot_action" if include_robot_action else "hidden_only",
            "pca_components": int(pca[1].shape[1]),
            "feature_dimension": int(x.shape[1]),
            "train_horizon": train_horizon,
            "evaluation_horizon": evaluation_horizon,
        },
        "counts": {
            "train": int(train.sum()), "calibration": int(calibration.sum()),
            "development": int(development.sum()),
        },
        "calibration": {
            "base_advantage_margin": margin,
            **calibration_summary,
        },
        "development_raw": evaluate_choices(
            raw_choice, data["outcomes"][development], data["utility"][development],
            data["sources"][development],
        ),
        "development_calibrated": evaluate_choices(
            calibrated_choice, data["outcomes"][development],
            data["utility"][development], data["sources"][development],
        ),
    }
    if evaluation_horizon is None:
        report["development_by_condition"] = evaluate_method_by_group(
            calibrated_choice, data, dev_indices, data["conditions"]
        )
        report["development_by_horizon"] = evaluate_method_by_group(
            calibrated_choice, data, dev_indices, data["horizons"]
        )
    return report


def fixed_evaluations(data: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    outcomes = data["outcomes"][mask]
    utility = data["utility"][mask]
    sources = data["sources"][mask]
    count = int(mask.sum())
    methods = {
        option: evaluate_choices(np.full(count, i), outcomes, utility, sources)
        for i, option in enumerate(OPTIONS)
    }
    oracle = np.argmax(utility, axis=1)
    methods["counterfactual_oracle"] = evaluate_choices(oracle, outcomes, utility, sources)
    return methods


def risk_gate_baseline(
    data: Mapping[str, Any], x: np.ndarray, ridge_alpha: float
) -> dict[str, Any]:
    train = data["splits"] == "train"
    calibration = data["splits"] == "calibration"
    development = data["splits"] == "development"
    target = (data["outcomes"][:, 0] == "catastrophe").astype(np.float64)
    model = fit_weighted_ridge(
        x[train], target[train], data["sources"][train], ridge_alpha,
    )
    scores = predict_ridge(model, x)
    threshold, option, calibration_summary = calibrate_risk_gate(
        scores[calibration], data["utility"][calibration], data["sources"][calibration]
    )
    choice = np.where(scores[development] > threshold, option, 0)
    return {
        "supervision": "binary_base_catastrophe_then_one_fixed_option",
        "fixed_intervention_option": OPTIONS[option],
        "calibration_threshold": threshold,
        "calibration": calibration_summary,
        "development": evaluate_choices(
            choice, data["outcomes"][development], data["utility"][development],
            data["sources"][development],
        ),
    }


def train_and_evaluate(
    root: Path, *, pca_components: int, ridge_alpha: float, catastrophe_cost: float
) -> dict[str, Any]:
    data = load_capture(root, catastrophe_cost)
    train = data["splits"] == "train"
    shared_pca = fit_frame_pca(
        data["arrays"]["hidden"], data["arrays"]["history_mask"], train,
        pca_components,
    )
    primary_x = build_features(
        data["arrays"], shared_pca, history=True, include_robot_action=True
    )
    variants = {}
    for history, robot_action in ((True, True), (False, True), (True, False), (False, False)):
        name = (
            ("history" if history else "single_frame") + "__" +
            ("hidden_robot_action" if robot_action else "hidden_only")
        )
        variants[name] = train_variant(
            data, history=history, include_robot_action=robot_action,
            pca_components=pca_components, ridge_alpha=ridge_alpha,
        )
    variants["h20_only__history__hidden_robot_action"] = train_variant(
        data, history=True, include_robot_action=True, pca_components=pca_components,
        ridge_alpha=ridge_alpha, train_horizon=20, evaluation_horizon=20,
    )
    variants["multi_h__history__hidden_robot_action__eval_h20"] = train_variant(
        data, history=True, include_robot_action=True, pca_components=pca_components,
        ridge_alpha=ridge_alpha, evaluation_horizon=20,
    )
    development = data["splits"] == "development"
    return {
        "schema_version": 1,
        "kind": "minimal_counterfactual_router_evaluation",
        "capture_root": str(root),
        "source_artifact_sha256": {
            name: _sha256(root / name) for name in (
                "capture_manifest.json", "decision_metadata.json",
                "option_rollouts.jsonl", "decision_features.npz",
            )
        },
        "protocol": {
            "utility": data["utility_map"],
            "model": "train-only shared frame PCA plus source-balanced linear ridge",
            "pca_components": pca_components,
            "ridge_alpha": ridge_alpha,
            "calibration": "one Base-favoring intervention margin",
            "development_policy": "evaluated once after calibration",
            "metadata_not_used_as_features": ["condition", "horizon", "placement", "split"],
        },
        "source_counts": {
            split: len(set(data["sources"][data["splits"] == split]))
            for split in ("train", "calibration", "development")
        },
        "development_fixed_methods": fixed_evaluations(data, development),
        "binary_risk_baseline": risk_gate_baseline(data, primary_x, ridge_alpha),
        "counterfactual_router_variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--catastrophe-cost", type=float, default=5.0)
    args = parser.parse_args()
    if args.pca_components < 1 or args.ridge_alpha < 0 or args.catastrophe_cost <= 0:
        raise SystemExit("PCA components and catastrophe cost must be positive")
    result = train_and_evaluate(
        args.capture.resolve(), pca_components=args.pca_components,
        ridge_alpha=args.ridge_alpha, catastrophe_cost=args.catastrophe_cost,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
