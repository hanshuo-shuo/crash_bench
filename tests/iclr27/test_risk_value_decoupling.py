from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.iclr27 import synthesize_risk_value_decoupling as synthesis


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/iclr27/risk_value_decoupling.yaml"


@pytest.fixture(scope="module")
def frozen_analysis() -> dict:
    # Analysis-only entry point: it validates and computes tables but never plots
    # or creates an output directory.
    return synthesis.analyze_frozen_inputs(CONFIG_PATH)


def test_canonical_labels_freeze_risk_benefit_and_base_first_ties() -> None:
    utility = np.asarray([
        [-1.0, 0.0, -1.0],  # risky and beneficial: Detour
        [0.0, 1.0, -1.0],   # safe and beneficial: Detour
        [1.0, 1.0, 0.0],    # no benefit; Base wins the equality
        [-1.0, -1.0, -1.0], # all-catastrophe no-good row; O still Base
        [0.0, 1.0, 1.0],    # D/R tie, Detour wins canonical O
    ])
    labels = synthesis.canonical_labels(
        utility,
        ["catastrophe", "safe_noncompletion", "task_success", "catastrophe", "safe_noncompletion"],
    )

    np.testing.assert_array_equal(labels["risk"], [1, 0, 0, 1, 0])
    np.testing.assert_array_equal(labels["benefit"], [1, 1, 0, 0, 1])
    np.testing.assert_array_equal(labels["optimal"], [1, 1, 0, 0, 1])
    np.testing.assert_array_equal(labels["intervention_advantage"], [1, 1, 0, 0, 1])
    np.testing.assert_array_equal(labels["detour_retreat_advantage"], [1, 2, 1, 0, 0])
    assert labels["strict_label"].tolist() == [
        "strict_detour", "strict_detour", "base_tie", "no_good_option",
        "detour_retreat_tie",
    ]
    with pytest.raises(ValueError, match="epsilon"):
        synthesis.canonical_labels(utility, ["catastrophe"] * 5, epsilon=1e-6)


def test_source_macro_is_equal_mean_over_source_level_rates() -> None:
    values = np.asarray([1.0, 1.0, 0.0, 0.0])
    sources = np.asarray(["dense", "dense", "dense", "sparse"])

    macro, count = synthesis.source_macro_mean(values, sources)
    conditional, conditional_count = synthesis.source_macro_mean(
        values, sources, denominator=np.asarray([True, False, False, True])
    )

    assert macro == pytest.approx(((2 / 3) + 0.0) / 2)
    assert count == 2
    assert conditional == pytest.approx(0.5)
    assert conditional_count == 2


def test_witness_selection_is_nearest_and_deterministic() -> None:
    metadata = [
        {"decision_id": "a", "source_state_sha256": "s", "condition": "glass", "horizon_actions": 5, "placement_id": "p0"},
        {"decision_id": "b", "source_state_sha256": "s", "condition": "glass", "horizon_actions": 5, "placement_id": "p1"},
        {"decision_id": "c", "source_state_sha256": "s", "condition": "glass", "horizon_actions": 5, "placement_id": "p2"},
        {"decision_id": "d", "source_state_sha256": "s", "condition": "glass", "horizon_actions": 5, "placement_id": "p3"},
    ]
    utility = np.asarray([
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
    ])
    outcomes = np.asarray([
        ["task_success", "safe_noncompletion", "catastrophe"],
        ["safe_noncompletion", "task_success", "catastrophe"],
        ["task_success", "safe_noncompletion", "catastrophe"],
        ["safe_noncompletion", "task_success", "catastrophe"],
    ])
    labels = synthesis.canonical_labels(utility, outcomes[:, 0])
    arrays = {
        "history_mask": np.ones((4, 1)),
        "robot_state": np.asarray([0.0, 0.1, 10.0, 20.0]).reshape(4, 1, 1),
        "nominal_action": np.asarray([0.0, 0.1, 10.0, 20.0]).reshape(4, 1, 1),
        "hidden": np.asarray([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]]).reshape(4, 1, 2),
    }

    first = synthesis.select_witnesses(metadata, labels, arrays, outcomes, utility)
    second = synthesis.select_witnesses(metadata, labels, arrays, outcomes, utility)

    assert first == second
    assert len(first) == 1
    assert (first[0]["state_a_id"], first[0]["state_b_id"]) == ("a", "b")
    assert first[0]["unordered_contrast"] == "Base_vs_Detour"


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({
            "risk_benefit_disagreement_sources": 2,
            "exact_same_risk_option_flip_sources": 2,
            "independent_mechanical_family_count": 2,
            "exact_flip_mechanical_family_count": 2,
            "glass_disagreement_sources": 0,
            "glass_source_profile_option_flip_sources": 0,
            "glass_exact_same_risk_option_flip_sources": 0,
        }, "BROAD_BENCHMARK_PIVOT"),
        ({
            "risk_benefit_disagreement_sources": 20,
            "exact_same_risk_option_flip_sources": 20,
            "independent_mechanical_family_count": 1,
            "exact_flip_mechanical_family_count": 1,
            "glass_disagreement_sources": 2,
            "glass_source_profile_option_flip_sources": 2,
            "glass_exact_same_risk_option_flip_sources": 1,
        }, "GLASS_SCOPED_BENCHMARK_PIVOT"),
        ({
            "risk_benefit_disagreement_sources": 1,
            "exact_same_risk_option_flip_sources": 1,
            "independent_mechanical_family_count": 1,
            "exact_flip_mechanical_family_count": 1,
            "glass_disagreement_sources": 1,
            "glass_source_profile_option_flip_sources": 1,
            "glass_exact_same_risk_option_flip_sources": 1,
        }, "INSUFFICIENT_FOR_PIVOT"),
    ],
)
def test_story_decision_has_all_three_literal_branches(counts: dict, expected: str) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert synthesis.decide_story(counts, config["story_rule"]) == expected


def test_policy_metrics_and_recovery_formula_report_raw_and_macro() -> None:
    outcomes = np.asarray([
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["task_success", "safe_noncompletion", "catastrophe"],
    ])
    utility = np.where(
        outcomes == "task_success", 1.0,
        np.where(outcomes == "catastrophe", -1.0, 0.0),
    )
    choice = np.asarray([1, 0, 0])
    metrics = synthesis.policy_metrics(choice, outcomes, utility, np.asarray(["dense", "dense", "sparse"]))

    assert metrics["raw_utility"] == pytest.approx(1 / 3)
    assert metrics["source_macro_utility"] == pytest.approx(0.5)
    assert metrics["raw_intervention_rate"] == pytest.approx(1 / 3)
    assert metrics["source_macro_intervention_rate"] == pytest.approx(0.25)
    assert synthesis.recovered_fraction(0.4, 0.2, 0.6) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Oracle"):
        synthesis.recovered_fraction(0.2, 0.2, 0.2)


def test_current_frozen_crosstab_and_waterfall_without_pdfs(frozen_analysis: dict) -> None:
    cells = {
        row["metric"]: row for row in frozen_analysis["risk_benefit_crosstab"]
        if row["row_type"] == "cell"
    }
    assert {key: cells[key]["raw_count"] for key in cells} == {
        "R0_B0": 172, "R0_B1": 9, "R1_B0": 10, "R1_B1": 82,
    }
    assert cells["R0_B0"]["source_macro_rate"] == pytest.approx(0.5830555555555554)
    waterfall = {row["stage"]: row for row in frozen_analysis["oracle_hybrid_waterfall"]}
    assert waterfall["Risk -> Best Fixed"]["source_macro_utility"] == pytest.approx(0.2544444444444444)
    assert waterfall["LearnedGate + OracleChoice"]["source_macro_utility"] == pytest.approx(0.2713888888888889)
    assert waterfall["OracleGate + learned linear choice"]["source_macro_utility"] == pytest.approx(0.4975)
    assert waterfall["OracleGate + learned tiny nonlinear choice"]["source_macro_utility"] == pytest.approx(0.4916666666666667)
    assert waterfall["full learned ADR"]["source_macro_utility"] == pytest.approx(0.2311111111111111)
    assert waterfall["OracleGate + OracleChoice"]["source_macro_utility"] == pytest.approx(0.5641666666666666)
    first = frozen_analysis["oracle_hybrid_waterfall"][0]
    assert first["oracle_to_tiny_choice_utility_loss"] == pytest.approx(0.0725)
    assert first["oracle_to_learned_gate_utility_loss"] == pytest.approx(0.29277777777777775)


def test_current_frozen_story_is_glass_scoped_and_keeps_51_of_60_visible(frozen_analysis: dict) -> None:
    story = frozen_analysis["story_decision"]
    counts = story["exact_independent_source_counts"]

    assert story["story_status"] == "GLASS_SCOPED_BENCHMARK_PIVOT"
    assert story["exactly_one_next_action"] == "authorize only a small non-glass option-ambiguity authoring screen"
    assert counts["risk_benefit_disagreement_sources"] == 12
    assert counts["glass_disagreement_sources"] == 6
    assert counts["glass_strict_retreat_sources"] == 9
    assert counts["glass_source_profile_option_flip_sources"] == 2
    assert counts["glass_exact_same_risk_option_flip_sources"] == 1
    assert counts["condition_family_count"] == 3
    assert counts["independent_mechanical_family_count"] == 1
    assert story["dominant_family"]["literal_concentration"] == "51/60"
    assert story["dominant_family"]["concentration"] == pytest.approx(0.85)
    assert story["dominant_family"]["strict_detour_or_retreat_sources"] == 16
    assert story["dominant_family"]["all_strict_detour_or_retreat_sources"] == 17
    assert len(frozen_analysis["same_risk_different_decision_witnesses"]) == 3


def test_oracle_benefit_best_fixed_is_global_detour_not_foldwise(frozen_analysis: dict) -> None:
    policies = {row["policy"]: row for row in frozen_analysis["risk_only_policy_regret"]}
    oracle_gate = policies["Oracle Benefit Gate -> Best Fixed"]

    assert oracle_gate["source_macro_utility"] == pytest.approx(0.49)
    assert oracle_gate["source_macro_intervention_rate"] == pytest.approx(0.34277777777777774)
    assert oracle_gate["choice_provenance"] == "Phase 2.5A global Detour under canonical B"
    assert policies["Risk -> Best Fixed"]["choice_provenance"] == "Phase 2.5B frozen foldwise OOF"


def test_sequential_table_separates_opportunity_timing_and_base_collapse(
    frozen_analysis: dict,
) -> None:
    rows = frozen_analysis["sequential_realization_gap"]
    by_layer = {row["evidence_layer"]: row for row in rows}

    assert by_layer["matched_state_oracle_opportunity"]["known_recovery_support"] == 2
    fixed = by_layer["matched_state_learned_performance"]
    assert fixed["intervention_count"] == 0
    assert fixed["missed_known_recovery"] == 2
    direct = next(
        row for row in rows
        if row["method_or_diagnostic"] == "Direct recovery Router"
    )
    assert direct["false_interventions"] == 0
    assert direct["base_collapse"] is True


def test_analysis_entry_point_does_not_create_pdf_artifacts() -> None:
    before = set(ROOT.glob("results/iclr27/risk_value_decoupling_*"))
    analysis = synthesis.analyze_frozen_inputs(CONFIG_PATH)
    after = set(ROOT.glob("results/iclr27/risk_value_decoupling_*"))

    assert analysis["config"]["outputs"]["required"] == list(synthesis.REQUIRED_OUTPUTS)
    assert after == before
    assert (
        analysis["evidence_package_fingerprint"]
        != analysis["config"]["inputs"]["capture_data_fingerprint"]
    )
