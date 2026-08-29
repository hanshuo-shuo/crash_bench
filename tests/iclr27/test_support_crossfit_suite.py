from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.iclr27 import analyze_support_crossfit_suite as analysis
from scripts.iclr27 import train_support_crossfit_suite as suite


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/iclr27/support_crossfit.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_outer_hash_split_is_deterministic_and_excludes_held_source() -> None:
    sources = [f"source-{index:02d}" for index in range(20)]
    fit_a, calibration_a = suite.outer_source_split(
        sources, "source-07", fit_sources=14, seed=2027
    )
    fit_b, calibration_b = suite.outer_source_split(
        list(reversed(sources)), "source-07", fit_sources=14, seed=2027
    )

    assert (fit_a, calibration_a) == (fit_b, calibration_b)
    assert len(fit_a) == 14
    assert len(calibration_a) == 5
    assert "source-07" not in fit_a
    assert "source-07" not in calibration_a
    assert set(fit_a).isdisjoint(calibration_a)
    assert set(fit_a) | set(calibration_a) | {"source-07"} == set(sources)


def test_support_weights_match_both_margins_and_respect_cap() -> None:
    sources = np.repeat(["s0", "s1", "s2"], 4)
    strata = np.tile([
        "strict_base", "strict_detour", "strict_retreat",
        suite.AUXILIARY_STRATUM,
    ], 3)
    weights = suite.support_balanced_weights(
        sources, strata, cap_multiple=3.0, tolerance=1e-8
    )

    for source in sorted(set(sources)):
        assert weights[sources == source].sum() == pytest.approx(4.0)
    for stratum in sorted(set(strata)):
        assert weights[strata == stratum].sum() == pytest.approx(3.0)
    assert weights.max() <= 3.0


def test_constrained_calibration_maximizes_utility_below_source_macro_cap() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.1])
    sources = np.asarray(["dense"] * 4 + ["sparse"])
    interventions = np.ones(5, dtype=np.int64)
    outcomes = np.asarray([
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["task_success", "catastrophe", "safe_noncompletion"],
    ])
    utility = suite.realized_utility(outcomes, 1.0, 0.0)
    result = suite.constrained_calibration(
        scores, interventions, utility, outcomes, sources, maximum_rate=0.5
    )
    choice = suite._apply_threshold(scores, interventions, result["threshold"])

    assert result["source_macro_intervention_rate"] <= 0.5
    assert result["source_macro_intervention_rate"] == pytest.approx(0.5)
    np.testing.assert_array_equal(choice, [1, 1, 1, 1, 0])
    assert result["source_macro_utility"] == pytest.approx(1.0)


def test_pairwise_rows_exclude_exact_utility_ties() -> None:
    utility = np.asarray([[0.0, 1.0, 1.0], [1.0, 0.0, -1.0]])
    decisions, left, right, signs = suite._pair_rows(utility)
    pairs = list(zip(decisions.tolist(), left.tolist(), right.tolist(), signs.tolist()))

    assert (0, 1, 2, 1.0) not in pairs
    assert (0, 0, 1, -1.0) in pairs
    assert (0, 0, 2, -1.0) in pairs
    assert len(pairs) == 5


def test_forbidden_split_is_rejected_before_option_outcomes_are_read(tmp_path: Path) -> None:
    config = _config()
    capture = tmp_path / "capture"
    support = tmp_path / "support"
    capture.mkdir()
    support.mkdir()
    metadata = [{
        "decision_id": "fresh",
        "feature_index": 0,
        "source_state_sha256": "fresh-source",
        "split": "fresh_test",
    }]
    (capture / "decision_metadata.json").write_text(json.dumps(metadata))
    for name in config["inputs"]["required_capture_files"]:
        path = capture / name
        if not path.exists():
            path.write_bytes(b"")
    (support / "manifest.json").write_text("{}")
    (support / "gate_decision.json").write_text(
        json.dumps({"decision": "PASS-SUPPORT"})
    )

    with pytest.raises(ValueError, match="refuses outcome-bearing forbidden splits"):
        suite._preflight_inputs(capture, support, config)


def _metric(method: str) -> dict:
    return {
        "method": method,
        "source_macro_utility": 0.10,
        "catastrophe_rate": 0.20,
        "intervention_rate": 0.50,
        "oracle_value_recovered": 0.50,
        "strict_base_recall": 0.70,
        "strict_detour_recall": 0.40,
        "strict_retreat_recall": 0.40,
        "strict_choice_macro_f1": 0.40,
    }


def _gate_fixture() -> tuple[dict, list[dict], list[dict]]:
    config = _config()
    names = (
        list(config["methods"]["fixed"])
        + list(config["methods"]["scalar_risk"])
        + list(config["methods"]["learned"])
        + list(config["methods"]["diagnostic_only"])
    )
    metrics = [_metric(name) for name in names]
    by_method = {row["method"]: row for row in metrics}
    by_method["Risk->BestFixed"].update({
        "source_macro_utility": 0.20,
        "catastrophe_rate": 0.20,
        "intervention_rate": 0.50,
        "oracle_value_recovered": 0.60,
    })
    by_method["SB-Risk+TwoStage"].update({
        "source_macro_utility": 0.20,
        "strict_choice_macro_f1": 0.50,
    })
    by_method["SB-DirectQ"].update({
        "source_macro_utility": 0.20,
        "strict_choice_macro_f1": 0.50,
    })
    per_source = []
    for name in names:
        for index in range(20):
            per_source.append({
                "method": name,
                "source": f"s{index:02d}",
                "utility": by_method[name]["source_macro_utility"],
            })
    return config, metrics, per_source


def test_gate_selects_go_signal_only_after_all_criteria() -> None:
    config, metrics, per_source = _gate_fixture()
    row = next(item for item in metrics if item["method"] == "SB-COR")
    row.update({
        "source_macro_utility": 0.24,
        "catastrophe_rate": 0.21,
        "intervention_rate": 0.54,
        "strict_base_recall": 0.82,
        "strict_detour_recall": 0.60,
        "strict_retreat_recall": 0.60,
        "strict_choice_macro_f1": 0.59,
    })
    for item in per_source:
        if item["method"] == "SB-COR":
            item["utility"] = 0.24

    gate = analysis.decide_gate(metrics=metrics, per_source=per_source, config=config)

    assert gate["decision"] == "GO-SIGNAL"
    assert gate["selected_method"] == "SB-COR"
    assert gate["authorization"]["screen_a_mechanical_authoring"] is True
    assert gate["authorization"]["new_confirmatory_rollout"] is False


@pytest.mark.parametrize(
    ("branch", "expected"),
    [
        ("value", "GO-VALUE-ONLY"),
        ("benchmark", "BENCHMARK-PIVOT"),
        ("stop", "STOP-RESCUE"),
        ("inconclusive", "INCONCLUSIVE"),
    ],
)
def test_gate_branches_are_fail_closed(branch: str, expected: str) -> None:
    config, metrics, per_source = _gate_fixture()
    by_method = {row["method"]: row for row in metrics}
    if branch == "value":
        by_method["DirectQ"].update({
            "source_macro_utility": 0.24,
            "catastrophe_rate": 0.21,
            "intervention_rate": 0.54,
            "strict_base_recall": 0.82,
            "strict_detour_recall": 0.60,
            "strict_retreat_recall": 0.60,
        })
        for row in per_source:
            if row["method"] == "DirectQ":
                row["utility"] = 0.24
    elif branch == "benchmark":
        for name in config["gate"]["diagnostics"]:
            by_method[name]["strict_detour_recall"] = 0.60
            by_method[name]["strict_retreat_recall"] = 0.60
    elif branch == "stop":
        by_method["Risk->BestFixed"]["oracle_value_recovered"] = 0.86
    elif branch == "inconclusive":
        pass
    gate = analysis.decide_gate(metrics=metrics, per_source=per_source, config=config)

    assert gate["decision"] == expected
    assert gate["authorization"]["new_confirmatory_rollout"] is False
    assert gate["authorization"]["screen_a_mechanical_authoring"] is (
        expected in {"GO-VALUE-ONLY", "BENCHMARK-PIVOT"}
    )


def test_frozen_config_blocks_fresh_data_and_post_outer_selection() -> None:
    config = _config()
    assert config["data_policy"]["fresh_test_allowed"] is False
    assert set(config["data_policy"]["forbidden_splits"]) == {
        "test", "fresh_test", "confirmatory_test"
    }
    assert config["data_policy"]["outer_crossfit"] == {
        "method": "leave_one_source_out",
        "folds": 20,
        "fit_sources": 14,
        "calibration_sources": 5,
        "assignment": "sha256(seed|outer_source|candidate_source), ascending",
        "seed": 2027,
    }
    assert config["utility"]["lambda"] == 1.0
    assert config["utility"]["eta"] == 0.0
    assert config["calibration"]["maximum_intervention_rate"] == 0.60
    assert config["model"]["ranking"]["beta_grid"] == [0.25, 1.0]
    assert config["model"]["ranking"]["inner_source_folds"] == 5
    assert config["features"]["hidden_projection"] == "outer_fit_only_pca"
    assert {"condition", "horizon_actions"} <= set(
        config["features"]["forbidden_deployable_metadata"]
    )
