"""Option identity, information, termination, and handoff contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


class CatalogKind(str, Enum):
    DEPLOYABLE = "deployable"
    DIAGNOSTIC_ORACLE = "diagnostic_oracle"


class HandoffMode(str, Enum):
    CONTINUE_OPTION = "continue_option"
    BASE_REQUERY_RESET = "base_requery_reset"
    SAFE_STOP = "safe_stop"


class TerminationReason(str, Enum):
    RUNNING = "running"
    GOAL_REACHED = "goal_reached"
    HAZARD_CLEAR = "hazard_clear"
    MAX_DURATION = "max_duration"
    SAFETY_STOP = "safety_stop"
    ADMISSIBILITY_REVOKED = "admissibility_revoked"
    EXECUTION_FAILURE = "execution_failure"


class InformationBoundaryViolation(PermissionError):
    """A deployable option attempted to read an unapproved or oracle field."""


class InformationView(Mapping[str, Any]):
    def __init__(self, payload: Mapping[str, Any], allowed_fields: frozenset[str]):
        self._payload = payload
        self._allowed = allowed_fields

    def __getitem__(self, key: str) -> Any:
        if key not in self._allowed:
            raise InformationBoundaryViolation(f"field {key!r} is outside the option information contract")
        if key not in self._payload:
            raise KeyError(key)
        return self._payload[key]

    def __iter__(self):
        return iter(sorted(self._allowed & self._payload.keys()))

    def __len__(self) -> int:
        return len(self._allowed & self._payload.keys())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._allowed and key in self._payload


@dataclass(frozen=True)
class OptionSpec:
    option_id: str
    version: int
    catalog_kind: CatalogKind
    mechanical_family: str
    max_duration_steps: int
    information_fields: frozenset[str]
    intervention_cost: float
    allows_return_to_base: bool
    snapshot_contract_version: int

    def __post_init__(self) -> None:
        if not self.option_id or self.version < 1:
            raise ValueError("option_id must be non-empty and version positive")
        if self.max_duration_steps < 1:
            raise ValueError("max_duration_steps must be positive")
        if self.intervention_cost < 0:
            raise ValueError("intervention_cost cannot be negative")
        if self.snapshot_contract_version < 1:
            raise ValueError("snapshot_contract_version must be positive")
        forbidden = {"future_outcome", "true_mechanism", "oracle_geometry", "task_success_predicate"}
        if self.catalog_kind == CatalogKind.DEPLOYABLE and forbidden & self.information_fields:
            raise InformationBoundaryViolation(
                f"deployable option {self.option_id} requests oracle fields: "
                f"{sorted(forbidden & self.information_fields)}"
            )

    @property
    def canonical_id(self) -> str:
        return f"{self.catalog_kind.value}:{self.option_id}:v{self.version}"

    def view(self, payload: Mapping[str, Any]) -> InformationView:
        return InformationView(payload, self.information_fields)


@runtime_checkable
class RuntimeOption(Protocol):
    spec: OptionSpec

    def admissible(self, view: InformationView) -> bool: ...

    def start(self, view: InformationView) -> None: ...

    def step(self, view: InformationView) -> Any: ...

    def termination_reason(self, view: InformationView) -> TerminationReason: ...

    def snapshot_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...
