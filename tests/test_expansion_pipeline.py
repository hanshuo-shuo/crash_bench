from __future__ import annotations

import pytest

from crashbench.governance.gates import (
    Criterion,
    CriterionClass,
    EvidenceState,
    GatePolicy,
    evaluate_gate,
)


def criterion(name, kind, actual, operator, threshold, state=EvidenceState.OBSERVED):
    return Criterion(name, kind, actual, operator, threshold, state)


def development_policy(scoped=True):
    return GatePolicy(
        stage_id="D2_SCREEN",
        confirmatory=False,
        test_outcomes_opened=False,
        allow_scoped_continuation=scoped,
        go_next_action="FREEZE_BROAD_PROTOCOL",
        scoped_next_action="CONTINUE_EXPLORATORY_WITH_SCOPED_CLAIMS",
        fail_next_action="STOP_OR_REPAIR_HARD_VALIDITY",
    )


def test_hard_failure_is_no_go_and_cannot_be_compensated():
    result = evaluate_gate(
        [
            criterion("restore", CriterionClass.HARD_VALIDITY, 0.94, ">=", 0.95),
            criterion("support", CriterionClass.CLAIM_SCOPE, 100, ">=", 3),
        ],
        development_policy(),
    )
    assert result["status"] == "NO_GO"
    assert result["hard_failures"] == ["restore"]
    assert result["non_compensatory"]


def test_development_claim_threshold_can_scope_continue_before_test():
    result = evaluate_gate(
        [
            criterion("restore", CriterionClass.HARD_VALIDITY, 1.0, "==", 1.0),
            criterion("two_winners", CriterionClass.CLAIM_SCOPE, 1, ">=", 2),
        ],
        development_policy(),
    )
    assert result["status"] == "SCOPED_CONTINUE"
    assert result["next_action"] == "CONTINUE_EXPLORATORY_WITH_SCOPED_CLAIMS"


def test_confirmatory_gate_cannot_enable_scoped_continuation():
    with pytest.raises(ValueError, match="confirmatory"):
        GatePolicy("D8", True, True, True, "go", "scoped", "fail")


def test_missing_evidence_is_inconclusive_fail_closed():
    result = evaluate_gate(
        [
            criterion(
                "restore", CriterionClass.HARD_VALIDITY, None, ">=", 0.95, EvidenceState.NOT_OBSERVED
            )
        ],
        development_policy(),
    )
    assert result["status"] == "INCONCLUSIVE_FAIL_CLOSED"
    assert result["missing_evidence"] == ["restore"]


def test_all_criteria_pass_is_go_with_stable_hash():
    criteria = [criterion("restore", CriterionClass.HARD_VALIDITY, 1.0, "==", 1.0)]
    first = evaluate_gate(criteria, development_policy())
    second = evaluate_gate(criteria, development_policy())
    assert first == second
    assert first["status"] == "GO"
