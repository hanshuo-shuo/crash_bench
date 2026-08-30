"""Deployable history-only options shared across expansion mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import CatalogKind, HandoffMode, InformationView, OptionSpec, TerminationReason


SAFE_STOP_SPEC = OptionSpec(
    "safe_stop", 1, CatalogKind.DEPLOYABLE, "global", 20,
    frozenset({"force_history", "gripper_command"}), 0.05, False, 1
)
BACKTRACK_REQUERY_SPEC = OptionSpec(
    "backtrack_requery", 1, CatalogKind.DEPLOYABLE, "global", 40,
    frozenset({"eef_history", "current_eef", "force_history", "gripper_command"}),
    0.05, True, 1
)
OBSERVATION_REFRESH_SPEC = OptionSpec(
    "observation_refresh", 1, CatalogKind.DEPLOYABLE, "observation_staleness", 1,
    frozenset({"policy_observation", "observation_queue"}), 0.05, True, 1
)
CORRECTIVE_REQUERY_SPEC = OptionSpec(
    "corrective_requery", 1, CatalogKind.DEPLOYABLE, "action_drift", 20,
    frozenset({"commanded_action_history", "executed_action_history", "base_proposed_action"}),
    0.05, True, 1
)


class SafeStopOption:
    spec = SAFE_STOP_SPEC

    def __init__(self, *, stable_force_threshold: float = 1.0, stable_steps: int = 3):
        self.threshold = float(stable_force_threshold)
        self.required_stable_steps = int(stable_steps)
        if self.threshold <= 0 or self.required_stable_steps < 1:
            raise ValueError("safe-stop force threshold and stable steps must be positive")
        self.steps = 0
        self.stable_count = 0

    def admissible(self, view: InformationView) -> bool:
        return len(view["force_history"]) > 0

    def start(self, view: InformationView) -> None:
        self.steps = 0
        self.stable_count = 0

    def step(self, view: InformationView) -> np.ndarray:
        force = float(view["force_history"][-1])
        self.stable_count = self.stable_count + 1 if force <= self.threshold else 0
        self.steps += 1
        action = np.zeros(7, dtype=np.float64)
        action[6] = float(view["gripper_command"])
        return action

    def termination_reason(self, view: InformationView) -> TerminationReason:
        if self.stable_count >= self.required_stable_steps:
            return TerminationReason.SAFETY_STOP
        if self.steps >= self.spec.max_duration_steps:
            return TerminationReason.MAX_DURATION
        return TerminationReason.RUNNING

    def snapshot_state(self) -> Mapping[str, Any]:
        return {"contract_version": 1, "steps": self.steps, "stable_count": self.stable_count}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("contract_version") != 1:
            raise ValueError("safe-stop snapshot version mismatch")
        self.steps = int(state["steps"])
        self.stable_count = int(state["stable_count"])


class BacktrackRequeryOption:
    spec = BACKTRACK_REQUERY_SPEC

    def __init__(self, *, history_offset: int = 3, gain: float = 5.0, max_delta: float = 0.05):
        self.history_offset = int(history_offset)
        self.gain = float(gain)
        self.max_delta = float(max_delta)
        if self.history_offset < 1 or self.gain <= 0 or self.max_delta <= 0:
            raise ValueError("backtrack parameters must be positive")
        self.target: np.ndarray | None = None
        self.steps = 0

    def admissible(self, view: InformationView) -> bool:
        history = np.asarray(view["eef_history"])
        return history.ndim == 2 and history.shape[1] == 3 and len(history) > self.history_offset

    def start(self, view: InformationView) -> None:
        if not self.admissible(view):
            raise ValueError("insufficient EEF history for backtrack")
        self.target = np.asarray(view["eef_history"], dtype=np.float64)[-1 - self.history_offset].copy()
        self.steps = 0

    def step(self, view: InformationView) -> np.ndarray:
        if self.target is None:
            raise RuntimeError("backtrack option was not started")
        current = np.asarray(view["current_eef"], dtype=np.float64)
        delta = np.clip((self.target - current) * self.gain, -self.max_delta, self.max_delta)
        action = np.zeros(7, dtype=np.float64)
        action[:3] = delta
        action[6] = float(view["gripper_command"])
        self.steps += 1
        return action

    def termination_reason(self, view: InformationView) -> TerminationReason:
        if self.target is None:
            return TerminationReason.EXECUTION_FAILURE
        if np.linalg.norm(self.target - np.asarray(view["current_eef"], dtype=np.float64)) <= 0.01:
            return TerminationReason.HAZARD_CLEAR
        if self.steps >= self.spec.max_duration_steps:
            return TerminationReason.MAX_DURATION
        return TerminationReason.RUNNING

    def handoff_mode(self, view: InformationView) -> HandoffMode:
        return (
            HandoffMode.BASE_REQUERY_RESET
            if self.termination_reason(view) == TerminationReason.HAZARD_CLEAR
            else HandoffMode.CONTINUE_OPTION
        )

    def snapshot_state(self) -> Mapping[str, Any]:
        return {
            "contract_version": 1,
            "target": None if self.target is None else self.target.copy(),
            "steps": self.steps,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("contract_version") != 1:
            raise ValueError("backtrack snapshot version mismatch")
        target = state.get("target")
        self.target = None if target is None else np.asarray(target, dtype=np.float64).copy()
        self.steps = int(state["steps"])


class ObservationRefreshOption:
    spec = OBSERVATION_REFRESH_SPEC

    def __init__(self):
        self.invoked = False

    def admissible(self, view: InformationView) -> bool:
        return view["observation_queue"] is not None

    def refresh(self, view: InformationView) -> Mapping[str, Any]:
        queue = view["observation_queue"]
        observation = view["policy_observation"]
        self.invoked = True
        return queue.refresh(observation)

    def handoff_mode(self) -> HandoffMode:
        return HandoffMode.BASE_REQUERY_RESET if self.invoked else HandoffMode.CONTINUE_OPTION

    def snapshot_state(self) -> Mapping[str, Any]:
        return {"contract_version": 1, "invoked": self.invoked}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("contract_version") != 1:
            raise ValueError("observation-refresh snapshot version mismatch")
        self.invoked = bool(state["invoked"])


class CorrectiveRequeryOption:
    spec = CORRECTIVE_REQUERY_SPEC

    def __init__(self, *, max_correction: float = 0.15):
        self.max_correction = float(max_correction)
        self.estimated_bias: np.ndarray | None = None
        self.invocations = 0

    def admissible(self, view: InformationView) -> bool:
        commanded = np.asarray(view["commanded_action_history"])
        executed = np.asarray(view["executed_action_history"])
        return commanded.ndim == executed.ndim == 2 and commanded.shape == executed.shape and commanded.shape[1] == 7 and len(commanded) >= 2

    def start(self, view: InformationView) -> None:
        if not self.admissible(view):
            raise ValueError("insufficient matched command-execution history")
        commanded = np.asarray(view["commanded_action_history"], dtype=np.float64)
        executed = np.asarray(view["executed_action_history"], dtype=np.float64)
        estimate = np.median(executed - commanded, axis=0)
        estimate[6] = 0
        self.estimated_bias = np.clip(estimate, -self.max_correction, self.max_correction)

    def step(self, view: InformationView) -> np.ndarray:
        if self.estimated_bias is None:
            raise RuntimeError("corrective requery option was not started")
        base = np.asarray(view["base_proposed_action"], dtype=np.float64)
        corrected = base - self.estimated_bias
        corrected[6] = base[6]
        self.invocations += 1
        return corrected

    def handoff_mode(self) -> HandoffMode:
        return HandoffMode.BASE_REQUERY_RESET

    def snapshot_state(self) -> Mapping[str, Any]:
        return {
            "contract_version": 1,
            "estimated_bias": None if self.estimated_bias is None else self.estimated_bias.copy(),
            "invocations": self.invocations,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("contract_version") != 1:
            raise ValueError("corrective-requery snapshot version mismatch")
        estimate = state.get("estimated_bias")
        self.estimated_bias = None if estimate is None else np.asarray(estimate, dtype=np.float64).copy()
        self.invocations = int(state["invocations"])
