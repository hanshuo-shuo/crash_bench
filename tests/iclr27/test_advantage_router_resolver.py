from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.iclr27 import run_advantage_router_resolver as resolver
from scripts.iclr27 import train_support_crossfit_suite as support_suite
from scripts.train_minimal_counterfactual_router import fit_frame_pca


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/iclr27/advantage_router_resolver.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_advantage_targets_use_the_frozen_algebra() -> None:
    utility = np.asarray([
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])

    a_int, a_dr = resolver.compute_advantages(utility)

    np.testing.assert_array_equal(a_int, [-1.0, 2.0, 1.0, 1.0])
    np.testing.assert_array_equal(a_dr, [1.0, 1.0, -2.0, 0.0])


def test_routing_rule_covers_all_sign_and_threshold_equalities() -> None:
    predicted_a_int = np.asarray([-1.0, 0.0, 0.1, 0.1, 1.0, 1.0])
    predicted_a_dr = np.asarray([-1.0, 1.0, -1.0, 0.0, -2.0, 2.0])

    choice = resolver.route_advantages(predicted_a_int, predicted_a_dr, tau=0.1)

    # score == tau remains Base; A_DR == 0 chooses Detour only after the gate opens.
    np.testing.assert_array_equal(choice, [0, 0, 0, 0, 2, 1])


def test_outer_source_never_enters_fit_or_calibration() -> None:
    sources = np.asarray([f"s{index:02d}" for index in range(20) for _ in range(2)])
    unique = sorted(set(sources.tolist()))
    fit_names, calibration_names = support_suite.outer_source_split(
        unique, "s07", fit_sources=14, seed=2027
    )

    held, fit, calibration = resolver.validate_outer_partition(
        sources, "s07", fit_names, calibration_names
    )

    assert held.sum() == 2
    assert fit.sum() == 28
    assert calibration.sum() == 10
    assert not np.any(held & (fit | calibration))
    with pytest.raises(ValueError, match="outer source entered"):
        resolver.validate_outer_partition(
            sources, "s07", [*fit_names, "s07"], calibration_names
        )


def test_pca_and_mlp_preprocessing_are_fit_fold_only() -> None:
    rng = np.random.default_rng(4)
    hidden = rng.normal(size=(8, 2, 6))
    mask = np.ones((8, 2), dtype=np.float64)
    fit = np.asarray([True, True, True, True, False, False, False, False])
    changed = hidden.copy()
    changed[~fit] += 10000.0
    pca_a = fit_frame_pca(hidden, mask, fit, components=3, seed=9)
    pca_b = fit_frame_pca(changed, mask, fit, components=3, seed=9)
    np.testing.assert_array_equal(pca_a[0], pca_b[0])
    np.testing.assert_array_equal(pca_a[1], pca_b[1])

    x = rng.normal(size=(8, 3))
    y = rng.normal(size=8)
    weights = np.ones(4)
    model_a, _ = resolver.fit_tiny_mlp(
        x[fit], y[fit], weights, hidden_units=3, l2=0.01,
        maximum_iterations=500, seed=11,
    )
    x_changed = x.copy()
    x_changed[~fit] -= 9999.0
    model_b, _ = resolver.fit_tiny_mlp(
        x_changed[fit], y[fit], weights, hidden_units=3, l2=0.01,
        maximum_iterations=500, seed=11,
    )
    for key in model_a:
        np.testing.assert_array_equal(model_a[key], model_b[key])


def test_oracle_targets_are_forbidden_from_deployable_features() -> None:
    config = _config()
    deployable = set(resolver.DEPLOYABLE_FEATURE_FIELDS)
    oracle = set(resolver.ORACLE_FIELDS)
    forbidden = set(config["features"]["forbidden_deployable_metadata"])

    assert deployable.isdisjoint(oracle)
    assert oracle <= forbidden
    assert set(config["features"]["full_future_free"]) == {
        "hidden_pca", "robot_state", "nominal_action"
    }


def test_calibration_enforces_the_source_macro_rho_cap_strictly() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    interventions = np.asarray([1, 1, 2, 2, 1, 2])
    sources = np.asarray(["dense", "dense", "dense", "dense", "sparse", "sparse"])
    outcomes = np.asarray([
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["catastrophe", "safe_noncompletion", "task_success"],
        ["task_success", "catastrophe", "safe_noncompletion"],
        ["task_success", "catastrophe", "safe_noncompletion"],
        ["task_success", "safe_noncompletion", "catastrophe"],
    ])
    utility = support_suite.realized_utility(outcomes, 1.0, 0.0)

    record = support_suite.constrained_calibration(
        scores, interventions, utility, outcomes, sources, maximum_rate=0.60
    )
    choice = support_suite._apply_threshold(scores, interventions, record["threshold"])

    rate = resolver._weighted_mean(choice != 0, sources)
    assert rate <= 0.60 + 1e-12
    assert record["source_macro_intervention_rate"] <= 0.60 + 1e-12


def test_same_seed_tiny_mlp_is_bitwise_deterministic() -> None:
    rng = np.random.default_rng(31)
    x = rng.normal(size=(14, 4))
    y = rng.normal(size=14)
    weights = np.linspace(0.5, 1.5, len(x))
    kwargs = dict(hidden_units=4, l2=0.02, maximum_iterations=500, seed=2027)

    model_a, summary_a = resolver.fit_tiny_mlp(x, y, weights, **kwargs)
    model_b, summary_b = resolver.fit_tiny_mlp(x, y, weights, **kwargs)

    assert summary_a == summary_b
    for key in model_a:
        np.testing.assert_array_equal(model_a[key], model_b[key])
    np.testing.assert_array_equal(
        resolver.predict_tiny_mlp(model_a, x), resolver.predict_tiny_mlp(model_b, x)
    )


def test_tiny_mlp_iteration_cap_returns_finite_deterministic_endpoint() -> None:
    rng = np.random.default_rng(52)
    x = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    weights = np.ones(len(x))

    model, summary = resolver.fit_tiny_mlp(
        x, y, weights, hidden_units=8, l2=0.01,
        maximum_iterations=1, seed=2027,
    )

    assert summary["converged"] is False
    assert summary["stopped_at_iteration_limit"] is True
    assert summary["iterations"] == 1
    assert np.all(np.isfinite(resolver.predict_tiny_mlp(model, x)))


@pytest.mark.parametrize(
    ("method_pass", "choice_pass", "upper_pass", "expected"),
    [
        ({"ADR-linear-split": True}, {}, {}, "METHOD_READY"),
        (
            {}, {"ADR-tiny-nonlinear-choice": True}, {},
            "CHOICE_CAPACITY_BOTTLENECK",
        ),
        ({}, {"ADR-linear-full": True}, {}, "GATE_OR_CALIBRATION_BOTTLENECK"),
        ({}, {}, {"ADR-linear-split": True}, "LABEL_MARGIN_BOTTLENECK"),
        ({}, {}, {}, "CURRENT_BENCHMARK_SIGNAL_INSUFFICIENT"),
    ],
)
def test_primary_decision_is_exhaustive_and_ordered(
    method_pass: dict, choice_pass: dict, upper_pass: dict, expected: str
) -> None:
    actual = resolver.select_primary_decision(
        method_pass=method_pass,
        choice_pass=choice_pass,
        upper_half_choice_pass=upper_pass,
    )
    assert actual == expected
    assert actual in resolver.PRIMARY_DECISIONS


def test_primary_decision_can_never_be_inconclusive() -> None:
    for method_bits in range(8):
        for choice_bits in range(8):
            for upper_bits in range(8):
                method = {
                    name: bool(method_bits & (1 << index))
                    for index, name in enumerate(resolver.ADR_METHODS)
                }
                choice = {
                    name: bool(choice_bits & (1 << index))
                    for index, name in enumerate(resolver.ADR_METHODS)
                }
                upper = {
                    name: bool(upper_bits & (1 << index))
                    for index, name in enumerate(resolver.ADR_METHODS)
                }
                decision = resolver.select_primary_decision(
                    method_pass=method, choice_pass=choice,
                    upper_half_choice_pass=upper,
                )
                assert decision in resolver.PRIMARY_DECISIONS
                assert decision != "INCONCLUSIVE"


def test_frozen_phase_2_5b_result_cannot_be_used_as_output(tmp_path: Path) -> None:
    frozen = tmp_path / "support_crossfit_frozen"
    frozen.mkdir()

    with pytest.raises(ValueError, match="refusing to overwrite frozen"):
        resolver.ensure_fresh_output(frozen, [frozen])
    with pytest.raises(ValueError, match="refusing to overwrite frozen"):
        resolver.ensure_fresh_output(frozen / "nested", [frozen])


def test_config_freezes_models_data_policy_and_one_action_per_decision() -> None:
    config = _config()
    assert tuple(config["models"]["candidates"]) == resolver.ADR_METHODS
    assert config["models"]["tiny_mlp"]["hidden_units"] == 32
    assert config["models"]["tiny_mlp"]["architecture_sweep"] is False
    assert config["data_policy"]["fresh_test_allowed"] is False
    assert config["calibration"]["maximum_intervention_rate"] == 0.60
    assert config["data_policy"]["outer_crossfit"]["fit_sources"] == 14
    assert config["data_policy"]["outer_crossfit"]["calibration_sources"] == 5
    assert set(config["decision"]["authorized_next_action"]) == set(
        resolver.PRIMARY_DECISIONS
    )
    assert all(
        isinstance(config["decision"]["authorized_next_action"][decision], str)
        and config["decision"]["authorized_next_action"][decision].strip()
        for decision in resolver.PRIMARY_DECISIONS
    )
