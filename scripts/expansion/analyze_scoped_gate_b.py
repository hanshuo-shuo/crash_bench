#!/usr/bin/env python3
"""Evaluate the prospectively scoped single-mechanism development Gate B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.utility import PhysicalBudgets
from crashbench.governance.gates import Criterion, CriterionClass, EvidenceState, GatePolicy, evaluate_gate
from crashbench.models.advantage import source_macro_policy_value
from crashbench.models.selective import PairwiseSourceConformalSelector
from scripts.expansion.calibrate_selector import MECHANISM_ID, predict_seed, sha256_file
from scripts.expansion.train_option_value import OPTION_IDS, build_training_arrays, read_jsonl


def source_macro_metric(
    rows: list[dict[str, Any]], choices: Mapping[str, str], key: str
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    expected_sources = {str(row["physical_source_id"]) for row in rows}
    for row in rows:
        if choices.get(str(row["block_id"])) == str(row["option_id"]):
            grouped[str(row["physical_source_id"])].append(float(row["outcome"][key]))
    if set(grouped) != expected_sources:
        raise ValueError("policy metric choices do not cover every development source")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def strict_recall(
    rows: list[dict[str, Any]], choices: Mapping[str, str], option_id: str, *, margin: float = 0.10
) -> tuple[float | None, int]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["block_id"])][str(row["option_id"])] = float(row["u0"])
    support = []
    for block, values in grouped.items():
        winner_value = values[option_id]
        if all(winner_value > value + margin for option, value in values.items() if option != option_id):
            support.append(block)
    if not support:
        return None, 0
    return sum(choices[block] == option_id for block in support) / len(support), len(support)


def analyze(
    *,
    branches: list[dict[str, Any]],
    data: Mapping[str, Any],
    utility_predictions: np.ndarray,
    catastrophe_predictions: np.ndarray,
    calibration: Mapping[str, Any],
    development_selection: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    development_rows = [row for row in branches if row["split_role"] == "development"]
    if set(data["roles"]) != {"development"}:
        raise ValueError("Gate B feature loader accessed a non-development role")
    selector = PairwiseSourceConformalSelector(
        OPTION_IDS,
        utility_quantiles=calibration["utility_pairwise_quantiles"],
        catastrophe_difference_quantiles=calibration["catastrophe_difference_quantiles"],
        catastrophe_absolute_quantiles=calibration["catastrophe_absolute_quantiles"],
    )
    mean_u = np.mean(utility_predictions, axis=0)
    mean_cat = np.mean(catastrophe_predictions, axis=0)
    blocks = np.asarray([str(row_id).rsplit(":", 1)[0] for row_id in data["row_ids"]])
    choices: dict[str, str] = {}
    decision_rows = []
    for block in sorted(set(blocks)):
        indices = np.flatnonzero(blocks == block)
        by_option = {str(data["options"][index]): int(index) for index in indices}
        predicted_utility = {option: float(mean_u[index]) for option, index in by_option.items()}
        predicted_catastrophe = {option: float(mean_cat[index]) for option, index in by_option.items()}
        scales = {}
        for option in OPTION_IDS:
            for other in OPTION_IDS:
                if option == other:
                    continue
                gaps = utility_predictions[:, by_option[option]] - utility_predictions[:, by_option[other]]
                scales[(option, other)] = max(float(np.std(gaps, ddof=1)), 0.05)
        selection = selector.select(
            mechanism_id=MECHANISM_ID,
            predicted_utility=predicted_utility,
            pairwise_scale=scales,
            predicted_catastrophe=predicted_catastrophe,
        )
        choices[block] = selection.option_id
        decision_rows.append(
            {
                "block_id": block,
                "option_id": selection.option_id,
                "action": selection.action,
                "minimum_pairwise_lcb": selection.minimum_pairwise_lcb,
                "catastrophe_difference_ucb": selection.catastrophe_difference_ucb,
                "catastrophe_absolute_ucb": selection.catastrophe_absolute_ucb,
            }
        )
    comparator = str(development_selection["frozen_strongest_deployable_comparator"])
    comparator_choices = baseline["development_choices"][comparator]
    model_value = source_macro_policy_value(development_rows, choices)
    comparator_value = source_macro_policy_value(development_rows, comparator_choices)
    model_cat = source_macro_metric(development_rows, choices, "catastrophe")
    comparator_cat = source_macro_metric(development_rows, comparator_choices, "catastrophe")
    recalls = {option: strict_recall(development_rows, choices, option) for option in OPTION_IDS}
    coverage = sum(option != "base_continue" for option in choices.values()) / len(choices)
    quantiles = [
        float(value)
        for key in (
            "utility_pairwise_quantiles",
            "catastrophe_difference_quantiles",
            "catastrophe_absolute_quantiles",
        )
        for value in calibration[key].values()
    ]
    hard = CriterionClass.HARD_VALIDITY
    claim = CriterionClass.CLAIM_SCOPE
    criteria = [
        Criterion("five_refit_models", hard, len(calibration["model_artifacts"]), "==", 5),
        Criterion("calibration_sources", hard, calibration["calibration_source_count"], ">=", 12),
        Criterion("finite_conformal_quantiles", hard, all(math.isfinite(value) for value in quantiles), "==", True),
        Criterion("calibration_test_rows_read", hard, calibration["test_rows_read"], "==", 0),
        Criterion("development_test_rows_read", hard, development_selection["test_rows_read"], "==", 0),
        Criterion("development_sources", hard, len(set(map(str, data["sources"]))), "==", 12),
        Criterion("seeds_beating_comparator", claim, development_selection["seeds_beating_comparator"], ">=", 4),
        Criterion("median_delta_u0", claim, development_selection["median_delta_u0"], ">=", 0.06),
        Criterion("calibrated_delta_u0", claim, model_value - comparator_value, ">", 0.0),
        Criterion("catastrophe_point_increase", claim, model_cat - comparator_cat, "<=", 0.02),
        Criterion("oracle_gap_recovered", claim, development_selection["oracle_gap_recovered"], ">=", 0.35),
        Criterion(
            "strict_base_recall", claim, recalls["base_continue"][0], ">=", 0.80,
            EvidenceState.OBSERVED if recalls["base_continue"][0] is not None else EvidenceState.NOT_OBSERVED,
        ),
        Criterion(
            "strict_refresh_recall", claim, recalls["observation_refresh"][0], ">=", 0.55,
            EvidenceState.OBSERVED if recalls["observation_refresh"][0] is not None else EvidenceState.NOT_OBSERVED,
        ),
        Criterion(
            "strict_safe_stop_recall", claim, recalls["safe_stop"][0], ">=", 0.55,
            EvidenceState.OBSERVED if recalls["safe_stop"][0] is not None else EvidenceState.NOT_OBSERVED,
        ),
        Criterion("intervention_coverage_lower", claim, coverage, ">=", 0.40),
        Criterion("intervention_coverage_upper", claim, coverage, "<=", 0.70),
    ]
    gate = evaluate_gate(
        criteria,
        GatePolicy(
            stage_id="D7_SCOPED_STALENESS_GATE_B",
            confirmatory=False,
            test_outcomes_opened=False,
            allow_scoped_continuation=True,
            go_next_action="AUTHORIZE_SCOPED_32_SOURCE_STATEWISE_METHOD_TEST",
            scoped_next_action="FREEZE_BENCHMARK_ONLY_MODE_AND_RUN_SCOPED_BENCHMARK_TEST",
            fail_next_action="STOP_BEFORE_TEST_AND_AUDIT_MODEL_CALIBRATION_VALIDITY",
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_scoped_gate_b_analysis",
        "scope": "single_primary_policy_single_staleness_mechanism_pilot",
        "prospective_adaptation": (
            "The original three-mechanism diversity clause is structurally inapplicable after D2; "
            "all model, safety, recall, coverage, and no-test-access criteria remain non-compensatory."
        ),
        "gate": gate,
        "frozen_strongest_deployable_comparator": comparator,
        "calibrated_source_macro_u0": model_value,
        "comparator_source_macro_u0": comparator_value,
        "delta_u0": model_value - comparator_value,
        "catastrophe_point_difference": model_cat - comparator_cat,
        "intervention_coverage": coverage,
        "strict_recall": {
            option: {"recall": value[0], "support_blocks": value[1]}
            for option, value in recalls.items()
        },
        "decisions": decision_rows,
        "test_rows_read": 0,
    }
    payload["analysis_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--refit-seed-dir", type=Path, action="append", required=True)
    parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument("--development-selection", type=Path, required=True)
    parser.add_argument("--baseline-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Gate B analysis: {args.output}")
    if len(args.refit_seed_dir) != 5:
        raise ValueError("Gate B requires exactly five refit seeds")
    calibration = json.loads(args.calibration_freeze.read_text())
    development_selection = json.loads(args.development_selection.read_text())
    baseline = json.loads(args.baseline_selection.read_text())
    if calibration["development_selection_sha256"] != sha256_file(args.development_selection):
        raise ValueError("development selection identity drift")
    if calibration["baseline_selection_sha256"] != sha256_file(args.baseline_selection):
        raise ValueError("baseline selection identity drift")
    utility = json.loads(args.utility_config.read_text())
    norm = utility["normalization"]
    budgets = PhysicalBudgets(
        norm["option_duration_steps"], norm["path_length_m"],
        norm["force_exposure_ns"], norm["latency_ms"], norm["source"]
    )
    anchors = read_jsonl(args.merged_dir / "anchors.jsonl")
    branches = read_jsonl(args.merged_dir / "branches.jsonl")
    data = build_training_arrays(
        anchors, branches, artifact_store=args.artifact_store, budgets=budgets,
        allowed_roles={"development"},
        feature_cache=args.feature_cache,
    )
    utility_predictions, catastrophe_predictions = [], []
    for expected_seed, directory in enumerate(args.refit_seed_dir):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("seed") != expected_seed or manifest.get("fit_on") != "train_development":
            raise ValueError("Gate B refit seed identity/order drift")
        predicted_u, predicted_cat = predict_seed(directory / "model.pt", data)
        utility_predictions.append(predicted_u)
        catastrophe_predictions.append(predicted_cat)
    payload = analyze(
        branches=branches,
        data=data,
        utility_predictions=np.asarray(utility_predictions),
        catastrophe_predictions=np.asarray(catastrophe_predictions),
        calibration=calibration,
        development_selection=development_selection,
        baseline=baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["gate"]["status"], "next_action": payload["gate"]["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
