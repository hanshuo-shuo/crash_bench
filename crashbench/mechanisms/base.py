"""Mechanism identity, matched controls, and pre-outcome validity contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class MechanicalValidity:
    valid: bool
    reason: str
    metrics: Mapping[str, float | int | bool | str]
    decided_before_option_outcome: bool = True

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("mechanical validity requires a reason")
        if not self.decided_before_option_outcome:
            raise ValueError("mechanical validity cannot depend on option outcomes")


@dataclass(frozen=True)
class MechanismSpec:
    mechanism_id: str
    version: int
    task_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    hazard_condition: str
    matched_control_conditions: tuple[str, ...]
    severity_ids: tuple[str, ...]
    deployable_option_ids: tuple[str, ...]
    diagnostic_option_ids: tuple[str, ...]
    information_contract_version: int

    def __post_init__(self) -> None:
        if not self.mechanism_id or self.version < 1:
            raise ValueError("mechanism identity and positive version are required")
        if len(set(self.task_ids)) != len(self.task_ids) or len(self.task_ids) < 1:
            raise ValueError("mechanism needs unique task IDs")
        if self.hazard_condition not in self.conditions:
            raise ValueError("hazard condition is absent from condition set")
        if not self.matched_control_conditions:
            raise ValueError("mechanism requires at least one matched control")
        if any(condition not in self.conditions for condition in self.matched_control_conditions):
            raise ValueError("matched control is absent from condition set")
        if self.hazard_condition in self.matched_control_conditions:
            raise ValueError("hazard condition cannot also be a matched control")
        if not self.severity_ids or len(set(self.severity_ids)) != len(self.severity_ids):
            raise ValueError("mechanism requires a unique frozen severity grid")
        if len(set(self.deployable_option_ids)) < 2:
            raise ValueError("mechanism requires at least two deployable options")
        overlap = set(self.deployable_option_ids) & set(self.diagnostic_option_ids)
        if overlap:
            raise ValueError(f"deployable and diagnostic option IDs overlap: {sorted(overlap)}")

    @property
    def canonical_id(self) -> str:
        return f"{self.mechanism_id}:v{self.version}"


@runtime_checkable
class Mechanism(Protocol):
    spec: MechanismSpec

    def prepare_injection(
        self, *, source: Mapping[str, Any], condition: str, severity_id: str
    ) -> Mapping[str, Any]: ...

    def validate_pre_outcome(
        self, *, source: Mapping[str, Any], injection: Mapping[str, Any]
    ) -> MechanicalValidity: ...

    def catastrophe_predicate(self, observation: Mapping[str, Any]) -> bool: ...
