"""Versioned observation-delay mechanism with exact queue continuation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import MechanicalValidity, MechanismSpec


OBSERVATION_STALENESS_SPEC = MechanismSpec(
    mechanism_id="observation_staleness",
    version=1,
    task_ids=("libero_spatial:0", "libero_spatial:2"),
    conditions=("stale", "fresh_control", "matched_buffer_control"),
    hazard_condition="stale",
    matched_control_conditions=("fresh_control", "matched_buffer_control"),
    severity_ids=("delay_1", "delay_3", "delay_5"),
    deployable_option_ids=("base_continue", "observation_refresh", "safe_stop"),
    diagnostic_option_ids=("oracle_fresh_observation",),
    information_contract_version=1,
)


def _copy_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in observation.items()}


class ObservationDelayQueue:
    def __init__(self, delay_steps: int):
        self.delay_steps = int(delay_steps)
        if self.delay_steps < 0:
            raise ValueError("observation delay cannot be negative")
        self._queue: deque[dict[str, np.ndarray]] = deque()
        self.observation_index = 0

    def push(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        current = _copy_observation(observation)
        self._queue.append(current)
        self.observation_index += 1
        if len(self._queue) > self.delay_steps + 1:
            self._queue.popleft()
        return _copy_observation(self._queue[0])

    def refresh(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        current = _copy_observation(observation)
        self._queue.clear()
        self._queue.append(current)
        return _copy_observation(current)

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "contract_version": 1,
            "delay_steps": self.delay_steps,
            "observation_index": self.observation_index,
            "queue": tuple(_copy_observation(row) for row in self._queue),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("contract_version") != 1:
            raise ValueError("observation-delay snapshot version mismatch")
        if int(state["delay_steps"]) != self.delay_steps:
            raise ValueError("observation-delay severity drift")
        self.observation_index = int(state["observation_index"])
        self._queue.clear()
        self._queue.extend(_copy_observation(row) for row in state["queue"])


def validate_staleness_pre_outcome(
    *, delay_steps: int, policy_action_period_ms: float, opportunity_slack_ms: float
) -> MechanicalValidity:
    metrics = {
        "delay_steps": int(delay_steps),
        "policy_action_period_ms": float(policy_action_period_ms),
        "delay_ms": float(delay_steps * policy_action_period_ms),
        "opportunity_slack_ms": float(opportunity_slack_ms),
    }
    if delay_steps < 1:
        return MechanicalValidity(False, "hazard_delay_must_be_positive", metrics)
    if policy_action_period_ms <= 0 or opportunity_slack_ms <= 0:
        return MechanicalValidity(False, "timing_budget_must_be_positive", metrics)
    if metrics["delay_ms"] >= opportunity_slack_ms:
        return MechanicalValidity(False, "delay_erases_entire_predeclared_opportunity", metrics)
    return MechanicalValidity(True, "staleness_window_mechanically_resolved", metrics)
