from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from crashbench.counterfactual_router import OPTIONS
from scripts.iclr27 import train_router_baseline_suite as suite


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/iclr27/baseline_suite.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_oracle_choice_prefers_base_then_cost_then_fixed_order() -> None:
    # Base wins an exact utility tie even if a caller assigns it a larger cost.
    choice = suite.oracle_choice_base_favoring(
        np.asarray([[1.0, 1.0, 0.0]]), [9.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(choice, [0])

    # With Base out of the tie, lower intervention cost wins.
    choice = suite.oracle_choice_base_favoring(
        np.asarray([[0.0, 1.0, 1.0]]), [0.0, 2.0, 1.0]
    )
    np.testing.assert_array_equal(choice, [2])

    # Equal-cost intervention ties use the frozen option order (Detour first).
    choice = suite.oracle_choice_base_favoring(
        np.asarray([[0.0, 1.0, 1.0]]), [0.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(choice, [1])


def test_rate_matching_is_source_balanced() -> None:
    # Four decisions from one source and one from another must not become 4:1
    # independent votes.  Selecting the dense source alone has macro rate 0.5.
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.1])
    sources = np.asarray(["dense", "dense", "dense", "dense", "sparse"])
    result = suite.calibrate_rate_threshold(scores, sources, 0.5)
    selected = scores > result["threshold"]
    source_rate = np.mean([
        np.mean(selected[sources == source]) for source in sorted(set(sources))
    ])

    assert result["source_balanced_rate"] == pytest.approx(0.5)
    assert source_rate == pytest.approx(0.5)
    assert np.mean(selected) == pytest.approx(0.8)


def test_best_fixed_is_selected_from_supplied_calibration_rows() -> None:
    risk_scores = np.asarray([0.9, 0.8, 0.1, 0.9, 0.1, 0.1])
    sources = np.asarray(["cal_a"] * 3 + ["cal_b"] * 3)
    outcomes = np.asarray([
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["task_success", "catastrophe", "catastrophe"],
        ["catastrophe", "safe_noncompletion", "catastrophe"],
        ["task_success", "catastrophe", "task_success"],
        ["task_success", "catastrophe", "task_success"],
    ])
    result = suite.select_best_fixed(
        risk_scores,
        outcomes,
        sources,
        threshold=0.5,
        utility_values={
            "task_success": 1.0,
            "safe_noncompletion": 0.0,
            "catastrophe": -1.0,
        },
    )

    assert result["option"] == "detour_complete"
    assert result["option_index"] == OPTIONS.index("detour_complete")


def test_source_split_leakage_is_rejected() -> None:
    suite.assert_source_disjoint(
        ["s0", "s0", "s1", "s2"],
        ["train", "train", "calibration", "development"],
    )
    with pytest.raises(ValueError):
        suite.assert_source_disjoint(
            ["s0", "s0", "s1"],
            ["train", "calibration", "development"],
        )


def test_multinomial_fit_is_finite_when_one_class_is_absent() -> None:
    x = np.asarray([
        [-1.0, 0.0], [-0.5, 0.1], [0.5, -0.1], [1.0, 0.0],
    ])
    labels = np.asarray([0, 0, 1, 1])  # class 2 is deliberately absent
    sources = np.asarray(["s0", "s0", "s1", "s1"])
    model, summary = suite.fit_multinomial(
        x, labels, sources, classes=3, l2=0.01
    )
    probabilities = suite.predict_multinomial(model, x)

    assert summary["converged"] is True
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def _write_test_only_capture(root: Path) -> None:
    metadata = [{
        "feature_index": 0,
        "decision_id": "d0",
        "source_state_sha256": "fresh-source",
        "split": "test",
        "condition": "glass",
        "horizon_actions": 20,
    }]
    (root / "decision_metadata.json").write_text(json.dumps(metadata))
    rollout_rows = [
        {"decision_id": "d0", "option": option, "outcome": outcome}
        for option, outcome in zip(
            OPTIONS,
            ("catastrophe", "task_success", "safe_noncompletion"),
        )
    ]
    (root / "option_rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rollout_rows)
    )
    np.savez_compressed(
        root / "decision_features.npz",
        hidden=np.zeros((1, 1, 2), dtype=np.float32),
        history_mask=np.ones((1, 1), dtype=np.float32),
        robot_state=np.zeros((1, 1, 8), dtype=np.float32),
        nominal_action=np.zeros((1, 1, 7), dtype=np.float32),
    )


def test_train_suite_refuses_outcome_bearing_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_test_only_capture(capture)

    def fail_if_outcomes_are_loaded(*_args, **_kwargs):
        raise AssertionError("load_capture must not inspect test outcomes")

    monkeypatch.setattr(suite, "load_capture", fail_if_outcomes_are_loaded)
    with pytest.raises(ValueError, match="refuses outcome-bearing test splits"):
        suite.train_suite(capture, _config(), tmp_path / "output")


def test_frozen_config_has_no_fresh_test_tuning() -> None:
    config = _config()
    protocol = config["protocol"]

    assert config["data_policy"]["allowed_splits"] == [
        "train", "calibration", "development"
    ]
    assert config["data_policy"]["fresh_test"]["allowed"] is False
    assert {"test", "fresh_test", "confirmatory_test"} <= set(
        protocol["forbidden_input_splits"]
    )
    assert protocol["primary_lambda"] == 1.0
    assert protocol["primary_method"] == "OutcomeRouter"
    assert protocol["noncompletion_cost"] == 0.0
    assert protocol["primary_target_intervention_rate"] == 0.6
    assert protocol["lambda_grid"] == [1.0, 2.0, 3.0, 5.0, 8.0]
    assert protocol["option_costs"] == [0.0, 1.0, 1.0]
    assert protocol["go_thresholds"] == {
        "intervention_rate_absolute": 0.10,
        "utility_absolute": 0.02,
        "catastrophe_rate_absolute": 0.02,
    }
    assert protocol["go_comparators"] == {
        "risk_reference": "Risk->BestFixed",
        "pivot_risk_methods": ["Risk->Detour", "Risk->BestFixed"],
        "go_a_direct_cover_methods": [
            "DirectChoice", "DirectQ", "PairwiseAdvantage"
        ],
        "go_b_value_methods": ["DirectQ", "PairwiseAdvantage"],
    }
    assert config["features"]["temporal_view"] == "single_frame"
    assert config["features"]["pca_components"] == 16
    assert config["features"]["source_weighting"] == "source_balanced"
    assert config["model"]["logistic_l2"] == 0.01
    assert config["model"]["ridge_alpha"] == 1.0
    assert config["statistics"]["bootstrap"] == {
        "unit": "source",
        "shared_resamples": True,
        "replicates": 5000,
        "seed": 2027,
    }
    geometry = next(row for row in config["baselines"] if row["id"] == "oracle_geometry")
    assert geometry["role"] == "diagnostic_only"
    assert geometry["eligible_for_go_no_go"] is False
