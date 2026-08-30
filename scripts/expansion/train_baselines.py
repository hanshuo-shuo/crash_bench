#!/usr/bin/env python3
"""Fit frozen deployable baselines and select the strongest on development only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.utility import PhysicalBudgets
from crashbench.models.advantage import RidgeOptionValue, best_fixed_option, source_macro_policy_value
from scripts.expansion.hash_tree_manifest import resolve_git_head
from scripts.expansion.train_option_value import (
    OPTION_IDS,
    build_training_arrays,
    read_jsonl,
)


def fit_logistic(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    iterations: int = 500,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
) -> np.ndarray:
    x = np.column_stack([np.ones(len(features)), np.asarray(features, dtype=np.float64)])
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(iterations):
        logits = np.clip(x @ coefficients, -30, 30)
        probability = 1 / (1 + np.exp(-logits))
        gradient = x.T @ (w * (probability - y)) / np.sum(w)
        gradient[1:] += l2 * coefficients[1:]
        coefficients -= learning_rate * gradient
    return coefficients


def logistic_probability(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(features)), np.asarray(features, dtype=np.float64)])
    logits = np.clip(x @ coefficients, -30, 30)
    return 1 / (1 + np.exp(-logits))


def choices_from_predictions(
    row_ids: np.ndarray,
    option_ids: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, str]:
    grouped = {}
    for row_id, option, prediction in zip(row_ids, option_ids, predictions):
        block = str(row_id).rsplit(":", 1)[0]
        grouped.setdefault(block, []).append((float(prediction), str(option)))
    return {
        block: max(sorted(values), key=lambda row: row[0])[1]
        for block, values in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite baseline output: {args.output_dir}")
    utility = json.loads(args.utility_config.read_text())
    norm = utility["normalization"]
    budgets = PhysicalBudgets(
        norm["option_duration_steps"], norm["path_length_m"],
        norm["force_exposure_ns"], norm["latency_ms"], norm["source"]
    )
    anchors = read_jsonl(args.merged_dir / "anchors.jsonl")
    branches = read_jsonl(args.merged_dir / "branches.jsonl")
    data = build_training_arrays(
        anchors,
        branches,
        artifact_store=args.artifact_store,
        budgets=budgets,
    )
    train = data["roles"] == "train"
    development = data["roles"] == "development"
    from crashbench.models.option_outcome import source_balanced_weights
    weights = source_balanced_weights(data["sources"][train])
    option_specific = RidgeOptionValue(OPTION_IDS, ridge=1.0, shared=False).fit(
        data["features"][train], data["options"][train], data["actual_u0"][train], weights
    )
    shared = RidgeOptionValue(OPTION_IDS, ridge=1.0, shared=True).fit(
        data["features"][train], data["options"][train], data["actual_u0"][train], weights
    )
    fixed, fixed_values = best_fixed_option(
        data["sources"][train], data["options"][train], data["actual_u0"][train]
    )
    nonbase = data["options"][train] != "base_continue"
    best_intervention, intervention_values = best_fixed_option(
        data["sources"][train][nonbase],
        data["options"][train][nonbase],
        data["actual_u0"][train][nonbase],
    )
    base_train = train & (data["options"] == "base_continue")
    risk_weights = source_balanced_weights(data["sources"][base_train])
    risk_coefficients = fit_logistic(
        data["features"][base_train],
        (data["outcomes"][base_train] == 1).astype(float),
        risk_weights,
    )
    dev_rows = [row for row in branches if row["split_role"] == "development"]
    dev_ids = data["row_ids"][development]
    dev_options = data["options"][development]
    dev_features = data["features"][development]
    predictions_specific = option_specific.predict(dev_features, dev_options)
    predictions_shared = shared.predict(dev_features, dev_options)
    choices = {
        "Base": {
            str(row_id).rsplit(":", 1)[0]: "base_continue" for row_id in dev_ids
        },
        "BestFixed": {
            str(row_id).rsplit(":", 1)[0]: fixed for row_id in dev_ids
        },
        "DirectQ": choices_from_predictions(dev_ids, dev_options, predictions_specific),
        "SharedQ": choices_from_predictions(dev_ids, dev_options, predictions_shared),
    }
    # Risk is block-level; every option row shares the same anchor feature.
    risk_choices = {}
    for block in sorted({str(row_id).rsplit(":", 1)[0] for row_id in dev_ids}):
        index = next(
            idx
            for idx, row_id in enumerate(dev_ids)
            if str(row_id).rsplit(":", 1)[0] == block
            and dev_options[idx] == "base_continue"
        )
        probability = float(logistic_probability(risk_coefficients, dev_features[index:index + 1])[0])
        risk_choices[block] = best_intervention if probability >= 0.5 else "base_continue"
    choices["RiskToBestFixed"] = risk_choices
    # Diagnostic oracle never enters comparator selection.
    oracle_choices = {}
    for row in dev_rows:
        current = oracle_choices.get(row["block_id"])
        if current is None or float(row["u0"]) > current[0]:
            oracle_choices[row["block_id"]] = (float(row["u0"]), row["option_id"])
    choices["OracleDiagnostic"] = {
        block: value[1] for block, value in oracle_choices.items()
    }
    metrics = {
        method: source_macro_policy_value(dev_rows, method_choices)
        for method, method_choices in choices.items()
    }
    deployable = {key: value for key, value in metrics.items() if key != "OracleDiagnostic"}
    comparator = max(sorted(deployable), key=lambda key: deployable[key])
    payload = {
        "schema_version": 1,
        "kind": "crashbench_expansion_frozen_baseline_selection",
        "git_commit": resolve_git_head(Path.cwd()),
        "train_rows": int(np.sum(train)),
        "development_rows": int(np.sum(development)),
        "calibration_rows_read": 0,
        "test_rows_read": 0,
        "best_fixed_option": fixed,
        "best_fixed_train_values": fixed_values,
        "best_fixed_intervention": best_intervention,
        "intervention_train_values": intervention_values,
        "development_source_macro_u0": metrics,
        "frozen_strongest_deployable_comparator": comparator,
        "risk_threshold": 0.5,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "baseline_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "baseline_models.npz",
        risk_coefficients=risk_coefficients,
        option_specific=np.asarray(
            [option_specific.coefficients_[option] for option in OPTION_IDS]
        ),
        shared=shared.coefficients_,
        option_ids=np.asarray(OPTION_IDS),
    )
    print(json.dumps({"output": str(args.output_dir), "comparator": comparator, "metrics": metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
