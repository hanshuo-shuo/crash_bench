"""Ordered, versioned mechanism registry and screen gate builder."""

from __future__ import annotations

from typing import Iterable, Mapping

from crashbench.governance.gates import (
    Criterion,
    CriterionClass,
    GatePolicy,
    evaluate_gate,
)

from .base import MechanismSpec


class MechanismRegistry:
    def __init__(self, specs: Iterable[MechanismSpec], *, ordered_ids: Iterable[str]):
        self._specs: dict[str, MechanismSpec] = {}
        for spec in specs:
            if spec.mechanism_id in self._specs:
                raise ValueError(f"duplicate mechanism_id: {spec.mechanism_id}")
            self._specs[spec.mechanism_id] = spec
        self.ordered_ids = tuple(ordered_ids)
        if set(self.ordered_ids) != set(self._specs) or len(self.ordered_ids) != len(self._specs):
            raise ValueError("ordered mechanism IDs must exactly match registered specs")

    def get(self, mechanism_id: str, *, version: int) -> MechanismSpec:
        try:
            spec = self._specs[mechanism_id]
        except KeyError as exc:
            raise KeyError(f"unknown mechanism: {mechanism_id}") from exc
        if spec.version != version:
            raise ValueError(
                f"mechanism version drift for {mechanism_id}: {version} != {spec.version}"
            )
        return spec

    def ordered_specs(self) -> tuple[MechanismSpec, ...]:
        return tuple(self._specs[key] for key in self.ordered_ids)


def evaluate_mechanism_screen(
    *,
    mechanism_id: str,
    summary: Mapping[str, float | int | bool],
    allow_scoped_continuation: bool = True,
) -> dict:
    hard = CriterionClass.HARD_VALIDITY
    claim = CriterionClass.CLAIM_SCOPE
    criteria = [
        Criterion("task0_nominal_success", hard, summary.get("task0_nominal_success"), ">=", 0.70),
        Criterion("task2_nominal_success", hard, summary.get("task2_nominal_success"), ">=", 0.70),
        Criterion("task0_eligible_sources", hard, summary.get("task0_eligible_sources"), ">=", 6),
        Criterion("task2_eligible_sources", hard, summary.get("task2_eligible_sources"), ">=", 6),
        Criterion("exact_restore_rate", hard, summary.get("exact_restore_rate"), ">=", 0.95),
        Criterion("admissible_execution_rate", hard, summary.get("admissible_execution_rate"), ">=", 0.90),
        Criterion("matched_control_catastrophe", hard, summary.get("matched_control_catastrophe"), "<=", 0.05),
        Criterion("mechanism_attribution_valid", hard, summary.get("mechanism_attribution_valid"), "==", True),
        Criterion("benefit_zero_sources", claim, summary.get("benefit_zero_sources"), ">=", 3),
        Criterion("benefit_one_sources", claim, summary.get("benefit_one_sources"), ">=", 3),
        Criterion("task0_benefit_sources", claim, summary.get("task0_benefit_sources"), ">=", 1),
        Criterion("task2_benefit_sources", claim, summary.get("task2_benefit_sources"), ">=", 1),
        Criterion("two_distinct_strict_winner_sources", claim, summary.get("two_distinct_strict_winner_sources"), ">=", 2),
    ]
    return evaluate_gate(
        criteria,
        GatePolicy(
            stage_id=f"D2_SCREEN:{mechanism_id}",
            confirmatory=False,
            test_outcomes_opened=False,
            allow_scoped_continuation=allow_scoped_continuation,
            go_next_action="ELIGIBLE_FOR_FORMAL_MECHANISM_SHORTLIST",
            scoped_next_action="MECHANICALLY_VALID_EXPLORATORY_ONLY__NO_BROAD_SUPPORT_CLAIM",
            fail_next_action="STOP_MECHANISM_VERSION",
        ),
    )
