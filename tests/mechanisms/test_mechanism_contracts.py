from __future__ import annotations

import pytest

from crashbench.mechanisms.base import MechanismSpec, MechanicalValidity
from crashbench.mechanisms.registry import MechanismRegistry, evaluate_mechanism_screen


def spec(mechanism_id="fragile"):
    return MechanismSpec(
        mechanism_id=mechanism_id,
        version=2,
        task_ids=("libero_spatial:0", "libero_spatial:2"),
        conditions=("hazard", "offpath", "no_hazard"),
        hazard_condition="hazard",
        matched_control_conditions=("offpath", "no_hazard"),
        severity_ids=("low", "high"),
        deployable_option_ids=("base_continue", "safe_stop", "backtrack_requery"),
        diagnostic_option_ids=("oracle_path",),
        information_contract_version=1,
    )


def passing_summary():
    return {
        "task0_nominal_success": 0.75,
        "task2_nominal_success": 0.75,
        "task0_eligible_sources": 6,
        "task2_eligible_sources": 6,
        "exact_restore_rate": 1.0,
        "admissible_execution_rate": 0.95,
        "matched_control_catastrophe": 0.0,
        "mechanism_attribution_valid": True,
        "benefit_zero_sources": 3,
        "benefit_one_sources": 3,
        "task0_benefit_sources": 1,
        "task2_benefit_sources": 1,
        "two_distinct_strict_winner_sources": 2,
    }


def test_mechanism_spec_requires_matched_controls_and_disjoint_option_ids():
    assert spec().canonical_id == "fragile:v2"
    with pytest.raises(ValueError, match="matched control"):
        MechanismSpec(
            "bad", 1, ("t",), ("hazard",), "hazard", (), ("s",), ("base", "stop"), (), 1
        )


def test_registry_pins_order_and_version():
    registry = MechanismRegistry([spec("a"), spec("b")], ordered_ids=("a", "b"))
    assert [row.mechanism_id for row in registry.ordered_specs()] == ["a", "b"]
    with pytest.raises(ValueError, match="version drift"):
        registry.get("a", version=1)


def test_screen_hard_failure_stops_version():
    summary = passing_summary()
    summary["exact_restore_rate"] = 0.90
    result = evaluate_mechanism_screen(mechanism_id="fragile", summary=summary)
    assert result["status"] == "NO_GO"
    assert result["next_action"] == "STOP_MECHANISM_VERSION"


def test_screen_support_shortfall_scopes_without_claiming_pass():
    summary = passing_summary()
    summary["two_distinct_strict_winner_sources"] = 1
    result = evaluate_mechanism_screen(mechanism_id="fragile", summary=summary)
    assert result["status"] == "SCOPED_CONTINUE"
    assert "two_distinct_strict_winner_sources" in result["claim_scope_failures"]


def test_screen_full_pass_is_shortlist_eligible():
    result = evaluate_mechanism_screen(mechanism_id="fragile", summary=passing_summary())
    assert result["status"] == "GO"


def test_mechanical_validity_must_be_pre_outcome():
    with pytest.raises(ValueError, match="cannot depend"):
        MechanicalValidity(True, "posthoc", {}, decided_before_option_outcome=False)
