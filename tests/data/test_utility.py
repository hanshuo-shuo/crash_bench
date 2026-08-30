from __future__ import annotations

import pytest

from crashbench.data.utility import (
    OutcomeVector,
    PhysicalBudgets,
    normalized_costs,
    option_decision,
    pareto_frontier,
    scalar_utility,
    sensitivity_weights,
)


def budgets(source="D1 physical control limits"):
    return PhysicalBudgets(100, 2.0, 50, 100, source)


def outcome(**overrides):
    values = {
        "task_success": 1,
        "catastrophe": 0,
        "safe_noncompletion": 0,
        "intervention_invoked": 1,
        "human_help": 0,
        "option_duration_steps": 50,
        "path_length_m": 1.0,
        "force_exposure_ns": 25,
        "latency_ms": 50,
    }
    values.update(overrides)
    return OutcomeVector(**values)


def test_primary_u0_and_clipped_physical_normalization():
    row = outcome(path_length_m=9.0)
    costs = normalized_costs(row, budgets())
    assert costs["path_length_norm"] == 1.0
    assert scalar_utility(row, budgets()) == pytest.approx(1 - 0.05 - 0.01 - 0.02 - 0.01 - 0.005)


def test_normalization_refuses_test_quantiles():
    with pytest.raises(ValueError, match="test quantiles"):
        budgets("95% test quantile")


def test_normalization_accepts_explicit_no_test_quantile_provenance():
    value = budgets(
        "physical control limits plus D1/D2 engineering timing; no formal/test quantiles"
    )
    assert value.source.endswith("no formal/test quantiles")


def test_outcome_terminal_indicators_are_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        outcome(task_success=1, catastrophe=1)


def test_full_predeclared_sensitivity_grid_has_108_unique_weights():
    grid = sensitivity_weights()
    assert len(grid) == 3 * 3 * 4 * 3 == 108
    assert len(set(grid)) == 108


def test_option_decision_preserves_ties_and_strict_gap():
    tied = option_decision(
        {"base": 0.2, "retreat": 0.4, "detour": 0.4},
        base_option_id="base",
        deployable_option_ids=("base", "retreat", "detour"),
    )
    assert tied.best_options == ("detour", "retreat")
    assert tied.strict_winner is None
    assert tied.benefit
    strict = option_decision(
        {"base": 0.2, "retreat": 0.31, "detour": 0.1},
        base_option_id="base",
        deployable_option_ids=("base", "retreat", "detour"),
        gamma=0.10,
    )
    assert strict.strict_winner == "retreat"
    assert strict.best_deployable_advantage == pytest.approx(0.11)


def test_pareto_frontier_keeps_tradeoffs_and_removes_dominated():
    assert pareto_frontier({"a": [1, 0], "b": [0, 1], "c": [0, 0]}) == ("a", "b")
