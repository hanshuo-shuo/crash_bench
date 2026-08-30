"""Variable-option realized branch rows and complete-decision validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .utility import OutcomeVector, PhysicalBudgets, UtilityWeights, scalar_utility


TERMINAL_BRANCH_STATUSES = frozenset(
    {"accepted", "execution_failure", "max_duration", "safety_stop"}
)


@dataclass(frozen=True)
class RealizedBranchRow:
    decision_id: str
    option_id: str
    branch_seed: int | str
    admissible: bool
    status: str
    outcome: OutcomeVector | None
    logical_attempt_key: str

    def __post_init__(self) -> None:
        if not self.decision_id or not self.option_id or not self.logical_attempt_key:
            raise ValueError("branch identity fields cannot be empty")
        if self.admissible:
            if self.status not in TERMINAL_BRANCH_STATUSES or self.outcome is None:
                raise ValueError("admissible branch needs a terminal status and outcome")
        elif self.status != "inadmissible" or self.outcome is not None:
            raise ValueError("inadmissible row must have status=inadmissible and no outcome")


@dataclass(frozen=True)
class DecisionDistribution:
    decision_id: str
    option_ids: tuple[str, ...]
    branch_seeds: tuple[int | str, ...]
    mean_utility: Mapping[str, float]
    catastrophe_count: Mapping[str, int]
    branch_count: Mapping[str, int]


def validate_complete_decision(
    rows: Iterable[RealizedBranchRow],
    *,
    admissible_option_ids: Iterable[str],
    declared_branch_seeds: Sequence[int | str],
) -> tuple[RealizedBranchRow, ...]:
    values = tuple(rows)
    if not values:
        raise ValueError("decision contains no branch rows")
    decision_ids = {row.decision_id for row in values}
    if len(decision_ids) != 1:
        raise ValueError("branch rows span multiple decision IDs")
    admissible = frozenset(admissible_option_ids)
    seeds = tuple(declared_branch_seeds)
    if not admissible or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("admissible options and unique declared branch seeds are required")
    seen: dict[tuple[str, int | str], RealizedBranchRow] = {}
    for row in values:
        key = (row.option_id, row.branch_seed)
        if key in seen:
            raise ValueError(f"duplicate realized branch row: {key}")
        seen[key] = row
        if row.option_id in admissible and not row.admissible:
            raise ValueError(f"planned admissible option is labeled inadmissible: {row.option_id}")
        if row.option_id not in admissible and row.admissible:
            raise ValueError(f"unplanned option appears admissible: {row.option_id}")
    expected = {(option, seed) for option in admissible for seed in seeds}
    actual = {key for key, row in seen.items() if row.admissible}
    missing = sorted(expected - actual, key=str)
    extra = sorted(actual - expected, key=str)
    if missing or extra:
        raise ValueError(f"incomplete realized-option decision: missing={missing}, extra={extra}")
    logical_keys = [seen[key].logical_attempt_key for key in expected]
    if len(set(logical_keys)) != len(logical_keys):
        raise ValueError("distinct planned branches reuse a logical attempt key")
    return tuple(seen[key] for key in sorted(expected, key=str))


def aggregate_decision_distribution(
    rows: Iterable[RealizedBranchRow],
    *,
    admissible_option_ids: Iterable[str],
    declared_branch_seeds: Sequence[int | str],
    budgets: PhysicalBudgets,
    weights: UtilityWeights = UtilityWeights(),
) -> DecisionDistribution:
    complete = validate_complete_decision(
        rows,
        admissible_option_ids=admissible_option_ids,
        declared_branch_seeds=declared_branch_seeds,
    )
    grouped: dict[str, list[RealizedBranchRow]] = {}
    for row in complete:
        grouped.setdefault(row.option_id, []).append(row)
    mean_utility = {}
    catastrophes = {}
    counts = {}
    for option, option_rows in sorted(grouped.items()):
        utilities = [scalar_utility(row.outcome, budgets, weights) for row in option_rows]
        mean_utility[option] = float(np.mean(utilities))
        catastrophes[option] = int(sum(row.outcome.catastrophe for row in option_rows))
        counts[option] = len(option_rows)
    return DecisionDistribution(
        decision_id=complete[0].decision_id,
        option_ids=tuple(sorted(grouped)),
        branch_seeds=tuple(declared_branch_seeds),
        mean_utility=mean_utility,
        catastrophe_count=catastrophes,
        branch_count=counts,
    )
