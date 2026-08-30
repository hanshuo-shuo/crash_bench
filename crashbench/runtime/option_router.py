"""Fail-closed monitor→option→termination→return runtime contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class RouterState(str, Enum):
    BASE = "BASE"
    OPTION_ACTIVE = "OPTION_ACTIVE"
    RETURN_PENDING = "RETURN_PENDING"
    SAFE_STOPPED = "SAFE_STOPPED"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class ReturnCertificate:
    hazard_clear: bool
    task_stage_valid: bool
    low_force_velocity_consecutive_observations: int
    option_state_serializable: bool
    base_catastrophe_ucb: float

    def passes(self, *, catastrophe_threshold: float = 0.10) -> bool:
        return (
            self.hazard_clear
            and self.task_stage_valid
            and self.low_force_velocity_consecutive_observations >= 3
            and self.option_state_serializable
            and 0.0 <= self.base_catastrophe_ucb <= catastrophe_threshold
        )


class OptionRouterRuntime:
    """Small deterministic state machine with an append-only causal trace."""

    def __init__(self, deployable_options: tuple[str, ...]):
        options = tuple(map(str, deployable_options))
        if "base_continue" not in options or "safe_stop" not in options:
            raise ValueError("runtime catalog must contain base_continue and safe_stop")
        if len(options) != len(set(options)):
            raise ValueError("runtime option catalog contains duplicates")
        self.deployable_options = options
        self.state = RouterState.BASE
        self.active_option: str | None = None
        self.trace: list[dict[str, Any]] = []

    def _record(self, event: str, **payload: Any) -> None:
        self.trace.append(
            {
                "event_index": len(self.trace),
                "event": event,
                "state": self.state.value,
                "active_option": self.active_option,
                **payload,
            }
        )

    def route(self, option_id: str, *, decision_metadata: Mapping[str, Any] | None = None) -> RouterState:
        option = str(option_id)
        if self.state != RouterState.BASE:
            raise RuntimeError("new routing decisions are only legal while Base is active")
        if option not in self.deployable_options:
            raise ValueError("router selected an option outside the deployable catalog")
        if option == "base_continue":
            self._record("PRESERVE_BASE", metadata=dict(decision_metadata or {}))
            return self.state
        if option == "safe_stop":
            self.state = RouterState.SAFE_STOPPED
            self._record("ABSTAIN_TO_SAFE_STOP", metadata=dict(decision_metadata or {}))
            return self.state
        self.active_option = option
        self.state = RouterState.OPTION_ACTIVE
        self._record("START_OPTION", metadata=dict(decision_metadata or {}))
        return self.state

    def request_return(self, certificate: ReturnCertificate) -> RouterState:
        if self.state != RouterState.OPTION_ACTIVE or self.active_option is None:
            raise RuntimeError("return may only be requested by an active option")
        if not certificate.passes():
            self.state = RouterState.SAFE_STOPPED
            self._record("RETURN_CERTIFICATE_FAILED", certificate=asdict(certificate))
            return self.state
        self.state = RouterState.RETURN_PENDING
        self._record("RETURN_CERTIFICATE_PASSED", certificate=asdict(certificate))
        return self.state

    def confirm_base_requery_reset(self, *, policy_state_synchronized: bool) -> RouterState:
        if self.state != RouterState.RETURN_PENDING:
            raise RuntimeError("Base requery/reset requires a passed return certificate")
        if not policy_state_synchronized:
            self.state = RouterState.SAFE_STOPPED
            self._record("BASE_SYNC_FAILED")
            return self.state
        returned_option = self.active_option
        self.active_option = None
        self.state = RouterState.BASE
        self._record("RETURN_TO_BASE", returned_option=returned_option)
        return self.state

    def terminate_episode(self, *, outcome: str) -> RouterState:
        if self.state == RouterState.TERMINAL:
            raise RuntimeError("episode is already terminal")
        previous = self.state
        self.active_option = None
        self.state = RouterState.TERMINAL
        self._record("EPISODE_TERMINAL", previous_state=previous.value, outcome=str(outcome))
        return self.state
