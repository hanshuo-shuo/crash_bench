from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.train_direct_recovery_window_head import (
    CANDIDATE_STATUS,
    CLASSES,
    FEATURE_DIMENSION,
    FIXED_L2,
    binary_auc,
    classification_metrics,
    fit_multinomial_head,
    leave_one_source_out_predictions,
    load_records,
    old_operational_predictions,
    old_scalar_advantage_predictions,
    predict_multinomial_head,
    run_experiment,
    source_balanced_weights,
)


def test_source_balanced_weights_equalize_source_mass():
    sources = np.asarray(["a", "a", "a", "b", "c", "c"])
    weights = source_balanced_weights(sources)

    assert np.isclose(weights.mean(), 1.0)
    mass = [weights[sources == source].sum() for source in ("a", "b", "c")]
    assert np.allclose(mass, mass[0])


def _separable_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    sources = []
    centers = np.zeros((len(CLASSES), FEATURE_DIMENSION), dtype=np.float64)
    centers[0, :2] = (-3.0, -1.0)
    centers[1, :2] = (3.0, -1.0)
    centers[2, :2] = (0.0, 3.0)
    for source_index in range(6):
        source_shift = (source_index - 2.5) * 0.02
        for class_index in range(len(CLASSES)):
            for repeat in range(2):
                value = centers[class_index].copy()
                value[2] = source_shift
                value[3] = repeat * 0.01
                features.append(value)
                labels.append(class_index)
                sources.append(f"source-{source_index}")
    return (
        np.asarray(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(sources),
    )


def test_multinomial_head_is_probabilistic_and_uses_fixed_shape():
    features, labels, sources = _separable_arrays()
    model, fit = fit_multinomial_head(features, labels, sources, l2=FIXED_L2)
    probabilities = predict_multinomial_head(model, features)

    assert fit["converged"] is True
    assert model["head_weight"].shape == (FEATURE_DIMENSION, len(CLASSES))
    assert model["head_bias"].shape == (len(CLASSES),)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(np.argmax(probabilities, axis=1) == labels) == 1.0


def test_loso_holds_every_row_from_a_source_in_one_fold():
    features, labels, sources = _separable_arrays()
    probabilities, fold_index, folds = leave_one_source_out_predictions(
        features, labels, sources
    )

    assert len(folds) == len(set(sources.tolist()))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    for source in set(sources.tolist()):
        source_folds = set(fold_index[sources == source].tolist())
        assert len(source_folds) == 1
        fold = folds[next(iter(source_folds))]
        assert fold["held_out_source_state_sha256"] == source
        assert source not in fold["training_sources"]


def test_old_operational_decision_uses_strict_frozen_margin():
    records = [
        {
            "old_router_advantage": 0.2,
            "old_router_pointwise_lcb": 0.1,
            "old_router_candidate_option": "Detour",
        },
        {
            "old_router_advantage": 0.1,
            "old_router_pointwise_lcb": 0.0,
            "old_router_candidate_option": "Detour",
        },
        {
            "old_router_advantage": 0.3,
            "old_router_pointwise_lcb": 0.2,
            "old_router_candidate_option": "FailSafeHold",
        },
        {
            "old_router_advantage": 0.0,
            "old_router_pointwise_lcb": -0.1,
            "old_router_candidate_option": "Detour",
        },
    ]
    prediction = old_operational_predictions(records)

    assert [CLASSES[index] for index in prediction] == [
        "Detour",
        "Base",
        "FailSafeHold",
        "Base",
    ]
    scalar_prediction = old_scalar_advantage_predictions(records)
    assert [CLASSES[index] for index in scalar_prediction] == [
        "Detour",
        "Detour",
        "FailSafeHold",
        "Base",
    ]


def test_metrics_use_half_credit_auc_ties_and_source_macro_accuracy():
    assert binary_auc(
        np.asarray([True, True, False, False]),
        np.asarray([1.0, 0.0, 1.0, 0.0]),
    ) == 0.5

    labels = np.asarray([0, 0, 0, 1])
    predictions = np.asarray([0, 0, 1, 1])
    sources = np.asarray(["large", "large", "large", "small"])
    metrics = classification_metrics(labels, predictions, sources)
    assert metrics["record_accuracy"] == 0.75
    assert np.isclose(metrics["source_macro_accuracy"], (2.0 / 3.0 + 1.0) / 2.0)
    assert metrics["class_metrics"]["FailSafeHold"]["predicted_count"] == 0


def _synthetic_records() -> list[dict]:
    records = []
    centers = {
        "Base": np.asarray([-3.0, -1.0] + [0.0] * 8),
        "Detour": np.asarray([3.0, -1.0] + [0.0] * 8),
        "FailSafeHold": np.asarray([0.0, 3.0] + [0.0] * 8),
    }
    placements = {
        "Base": "glass_recovery_heldout_0010",
        "Detour": "glass_recovery_heldout_0004",
        "FailSafeHold": "glass_recovery_heldout_0003",
    }
    for source_index in range(6):
        for class_index, option in enumerate(CLASSES):
            feature = centers[option].copy()
            feature[2] = (source_index - 2.5) * 0.02
            hard = option == "Base"
            recovery = option == "Detour"
            records.append({
                "decision_id": f"decision-{source_index}-{class_index}",
                "source_state_sha256": f"source-{source_index}",
                "placement_id": placements[option],
                "record_type": (
                    "hard_control_negative" if hard else "dense_glass_anchor"
                ),
                "condition": "glass" if not hard else "offpath",
                "preferred_option": option,
                "router_output_feature": feature.tolist(),
                # The following metadata is available for grouping/reporting but
                # never enters the feature matrix.
                "horizon_actions": 20,
                "action_index": 99 - source_index,
                "trajectory_id": f"trajectory-{source_index}",
                "base_outcome": "task_success" if hard else "catastrophe",
                "detour_outcome": "task_success" if recovery else None,
                "hold_outcome": (
                    "safe_noncompletion" if option == "FailSafeHold" else None
                ),
                "recovery_open": recovery,
                "intervention_needed": not hard,
                "hard_negative": hard,
                "loss_control": option == "FailSafeHold",
                "old_router_advantage": 1.0 if hard else 0.2,
                "old_router_pointwise_lcb": 0.1 if hard else -0.1,
                "old_router_candidate_option": (
                    "FailSafeHold" if option == "FailSafeHold" else "Detour"
                ),
            })
    return records


def test_end_to_end_writes_complete_oof_and_marked_candidate(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _synthetic_records()),
        encoding="utf-8",
    )
    report = tmp_path / "report.md"
    oof = tmp_path / "oof.csv"
    analysis = tmp_path / "analysis.json"
    confusion = tmp_path / "confusion.png"
    model = tmp_path / "model.npz"

    result = run_experiment(
        input_path,
        output_report=report,
        output_oof=oof,
        output_analysis=analysis,
        output_confusion=confusion,
        output_model=model,
    )

    assert result["counts"]["unique_sources"] == 6
    assert result["protocol"]["model_input_fields"] == ["router_output_feature"]
    assert all(path.exists() and path.stat().st_size > 0 for path in (
        report, oof, analysis, confusion, model
    ))
    with oof.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(_synthetic_records())
    assert all(
        np.isclose(sum(float(row[f"probability_{name}"]) for name in CLASSES), 1.0)
        for row in rows
    )
    saved_analysis = json.loads(analysis.read_text(encoding="utf-8"))
    assert saved_analysis["fold_integrity"][
        "all_rows_from_each_source_in_exactly_one_test_fold"
    ] is True
    with np.load(model) as archive:
        assert archive["status"].item() == CANDIDATE_STATUS
        assert archive["input_feature_field"].item() == "router_output_feature"
        assert archive["head_weight"].shape == (FEATURE_DIMENSION, len(CLASSES))


def test_loader_extracts_only_ten_dimensional_router_output(tmp_path: Path):
    records = _synthetic_records()
    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )

    loaded, features, labels, sources = load_records(path)

    assert loaded[0]["action_index"] != loaded[-1]["action_index"]
    assert features.shape == (len(records), FEATURE_DIMENSION)
    assert np.array_equal(features[0], np.asarray(records[0]["router_output_feature"]))
    assert labels.shape == (len(records),)
    assert sources.shape == (len(records),)
