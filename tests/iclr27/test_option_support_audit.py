from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.iclr27 import audit_option_support as audit


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/iclr27/option_support_audit.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_strict_labels_never_turn_ties_into_support() -> None:
    utility = np.asarray([
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [-1.0, -1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
    ])
    labels, margins = audit.strict_option_labels(utility)

    assert labels.tolist() == [
        "strict_base",
        "strict_detour",
        "strict_retreat",
        "detour_retreat_tie",
        "base_tie",
        "no_good_option",
        "strict_detour",
        "base_tie",
    ]
    np.testing.assert_allclose(margins, [1, 1, 1, 0, 0, 0, 1, 0])


def test_source_support_uses_gamma_and_excludes_ties() -> None:
    rows = [
        {"source": "a", "split": "train", "strict_label": "strict_base", "strict_advantage": 1.0},
        {"source": "a", "split": "train", "strict_label": "strict_retreat", "strict_advantage": 0.25},
        {"source": "b", "split": "calibration", "strict_label": "strict_retreat", "strict_advantage": 1.0},
        {"source": "b", "split": "calibration", "strict_label": "detour_retreat_tie", "strict_advantage": 0.0},
        {"source": "c", "split": "development", "strict_label": "strict_detour", "strict_advantage": 2.0},
        {"source": "c", "split": "development", "strict_label": "strict_base", "strict_advantage": 1.0},
    ]
    label_rows, per_source = audit.support_rows(rows, gamma=0.5)
    support = {row["strict_label"]: row["source_support"] for row in label_rows}

    assert support["strict_base"] == 2
    assert support["strict_detour"] == 1
    assert support["strict_retreat"] == 1
    assert support["detour_retreat_tie"] == 0
    source_c = next(row for row in per_source if row["source"] == "c")
    assert source_c["has_base_intervention_strict_flip"] is True
    assert source_c["qualifies_gate_flip"] is True


@pytest.mark.parametrize(
    ("retreat", "fixed", "expected"),
    [
        (9, 0.796, "PASS-SUPPORT"),
        (4, 0.82, "SUPPLEMENT-SUPPORT"),
        (2, 0.50, "STOP-SUPPORT"),
        (9, 0.91, "STOP-SUPPORT"),
        (9, 0.82, "INCONCLUSIVE"),
    ],
)
def test_gate_branches_are_fail_closed(
    retreat: int, fixed: float, expected: str
) -> None:
    result = audit.decide_gate(
        strict_base_support=16,
        strict_detour_support=13,
        strict_retreat_support=retreat,
        within_source_flip_sources=13,
        best_fixed_recovery=fixed,
        gate_config=_config()["gate"],
    )
    assert result["decision"] == expected
    assert result["authorization"]["phase_2_5b_pooled_crossfit"] is (
        expected == "PASS-SUPPORT"
    )
    assert result["authorization"]["screen_a_mechanical_authoring"] is False
    assert result["authorization"]["new_confirmatory_rollout"] is False


def test_loso_diagnostic_does_not_train_on_held_source() -> None:
    rows = [
        {"decision_id": "a", "source": "held", "condition": "glass", "horizon": 10, "strict_label": "strict_retreat"},
        {"decision_id": "b", "source": "train1", "condition": "glass", "horizon": 10, "strict_label": "strict_detour"},
        {"decision_id": "c", "source": "train2", "condition": "glass", "horizon": 20, "strict_label": "strict_detour"},
        {"decision_id": "d", "source": "train3", "condition": "noglass", "horizon": 10, "strict_label": "strict_base"},
    ]
    predictions = audit.diagnostic_prediction_rows(rows, ["condition"])
    held_three_way = next(
        row for row in predictions
        if row["decision_id"] == "a" and row["target_task"] == "strict_three_way"
    )
    held_two_way = next(
        row for row in predictions
        if row["decision_id"] == "a"
        and row["target_task"] == "strict_intervention_two_way"
    )

    assert held_three_way["prediction"] == "strict_detour"
    assert held_two_way["prediction"] == "strict_detour"
    assert held_three_way["correct"] is False


def test_risk_matching_is_caliper_bounded_and_without_replacement() -> None:
    rows = [
        {"decision_id": "d0", "source": "s0", "split": "train", "condition": "glass", "horizon": 20, "strict_label": "strict_detour", "base_catastrophe_risk": 0.10},
        {"decision_id": "d1", "source": "s1", "split": "train", "condition": "glass", "horizon": 10, "strict_label": "strict_detour", "base_catastrophe_risk": 0.80},
        {"decision_id": "r0", "source": "s2", "split": "calibration", "condition": "glass", "horizon": 20, "strict_label": "strict_retreat", "base_catastrophe_risk": 0.11},
        {"decision_id": "r1", "source": "s3", "split": "development", "condition": "glass", "horizon": 5, "strict_label": "strict_retreat", "base_catastrophe_risk": 0.83},
        {"decision_id": "r2", "source": "s4", "split": "development", "condition": "glass", "horizon": 5, "strict_label": "strict_retreat", "base_catastrophe_risk": 0.40},
    ]
    pairs = audit.matched_risk_pair_rows(rows, maximum_difference=0.05)

    assert len(pairs) == 2
    assert all(row["absolute_risk_difference"] <= 0.05 for row in pairs)
    assert len({row["detour_decision_id"] for row in pairs}) == len(pairs)
    assert len({row["retreat_decision_id"] for row in pairs}) == len(pairs)


def test_timing_transitions_do_not_bridge_over_a_tie() -> None:
    common = {
        "source": "s0",
        "placement_id": "p0",
        "condition": "glass",
    }
    rows = [
        {**common, "decision_id": "early", "horizon": 30, "matched_scan_index": 0, "strict_label": "strict_detour"},
        {**common, "decision_id": "tie", "horizon": 20, "matched_scan_index": 1, "strict_label": "detour_retreat_tie"},
        {**common, "decision_id": "late", "horizon": 10, "matched_scan_index": 2, "strict_label": "strict_retreat"},
    ]
    transitions = audit.transition_rows(rows)
    assert transitions == []


def test_fresh_or_test_split_is_rejected() -> None:
    config = _config()
    audit.validate_allowed_splits(["train", "calibration", "development"], config)
    with pytest.raises(ValueError, match="refuses outcome-bearing"):
        audit.validate_allowed_splits(
            ["train", "calibration", "development", "fresh_test"], config
        )


def test_frozen_config_keeps_primary_gate_definition() -> None:
    config = _config()
    assert config["utility"] == {
        "lambda": 1.0,
        "eta": 0.0,
        "outcome_values": {
            "task_success": 1.0,
            "safe_noncompletion": 0.0,
            "catastrophe": -1.0,
        },
        "option_order": list(audit.OPTIONS),
    }
    assert config["strict_labels"]["support_advantage_gamma"] == 0.5
    assert config["data_policy"]["expected_source_count"] == 20
    assert config["data_policy"]["expected_historical_split_source_counts"] == {
        "train": 5,
        "calibration": 7,
        "development": 8,
    }
    assert config["fixed_option_recovery"]["gate_aggregation"] == "decision_weighted"
    assert config["data_policy"]["fresh_test_allowed"] is False
    assert set(config["data_policy"]["forbidden_splits"]) == {
        "test", "fresh_test", "confirmatory_test"
    }
    assert config["gate"]["pass_support"] == {
        "strict_base_source_support_min": 8,
        "strict_detour_source_support_min": 6,
        "strict_retreat_source_support_min": 6,
        "within_source_flip_sources_min": 4,
        "best_fixed_nonbase_oracle_value_recovered_max": 0.80,
    }
