from __future__ import annotations

import pytest

from crashbench.data.realized_options import (
    RealizedBranchRow,
    aggregate_decision_distribution,
    validate_complete_decision,
)
from crashbench.data.utility import OutcomeVector, PhysicalBudgets


def success(intervention=0):
    return OutcomeVector(1, 0, 0, intervention, 0, 1, 0.1, 0, 2)


def catastrophe(intervention=0):
    return OutcomeVector(0, 1, 0, intervention, 0, 1, 0.1, 1, 2)


def row(option, seed, outcome, *, status="accepted", admissible=True):
    return RealizedBranchRow(
        decision_id="decision-1",
        option_id=option,
        branch_seed=seed,
        admissible=admissible,
        status=status,
        outcome=outcome,
        logical_attempt_key=f"{option}:{seed}",
    )


def test_variable_option_and_seed_distribution_is_preserved():
    rows = [
        row("base", 0, catastrophe()),
        row("base", 1, success()),
        row("retreat", 0, success(1)),
        row("retreat", 1, success(1)),
        row("refresh", 0, success(1)),
        row("refresh", 1, catastrophe(1), status="execution_failure"),
    ]
    result = aggregate_decision_distribution(
        rows,
        admissible_option_ids=("base", "retreat", "refresh"),
        declared_branch_seeds=(0, 1),
        budgets=PhysicalBudgets(100, 2, 50, 100, "physical limits"),
    )
    assert result.option_ids == ("base", "refresh", "retreat")
    assert result.branch_count == {"base": 2, "refresh": 2, "retreat": 2}
    assert result.catastrophe_count == {"base": 1, "refresh": 1, "retreat": 0}


def test_missing_planned_branch_fails_closed():
    rows = [row("base", 0, success()), row("retreat", 0, success(1))]
    with pytest.raises(ValueError, match="incomplete"):
        validate_complete_decision(
            rows,
            admissible_option_ids=("base", "retreat"),
            declared_branch_seeds=(0, 1),
        )


def test_inadmissible_option_does_not_create_required_branch():
    rows = [
        row("base", 0, success()),
        row("retreat", 0, None, status="inadmissible", admissible=False),
    ]
    complete = validate_complete_decision(
        rows, admissible_option_ids=("base",), declared_branch_seeds=(0,)
    )
    assert len(complete) == 1 and complete[0].option_id == "base"


def test_duplicate_branch_and_logical_key_reuse_are_rejected():
    duplicate = [row("base", 0, success()), row("base", 0, success())]
    with pytest.raises(ValueError, match="duplicate"):
        validate_complete_decision(
            duplicate, admissible_option_ids=("base",), declared_branch_seeds=(0,)
        )
    shared_key = row("retreat", 0, success(1))
    object.__setattr__(shared_key, "logical_attempt_key", "base:0")
    with pytest.raises(ValueError, match="reuse"):
        validate_complete_decision(
            [row("base", 0, success()), shared_key],
            admissible_option_ids=("base", "retreat"),
            declared_branch_seeds=(0,),
        )
