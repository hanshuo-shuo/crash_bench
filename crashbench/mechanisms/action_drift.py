"""Versioned command-execution drift mechanism with matched zero-bias control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import MechanicalValidity, MechanismSpec


ACTION_DRIFT_SPEC = MechanismSpec(
    mechanism_id="action_drift",
    version=1,
    task_ids=("libero_spatial:0", "libero_spatial:2"),
    conditions=("drift", "zero_bias_control", "orthogonal_bias_control"),
    hazard_condition="drift",
    matched_control_conditions=("zero_bias_control", "orthogonal_bias_control"),
    severity_ids=("bias_005", "bias_010", "bias_015"),
    deployable_option_ids=("base_continue", "corrective_requery", "safe_stop"),
    diagnostic_option_ids=("oracle_inverse_drift",),
    information_contract_version=1,
)


@dataclass(frozen=True)
class ActionDriftState:
    contract_version: int
    applied_steps: int
    cumulative_command_error: tuple[float, ...]


class ActionDriftInjector:
    def __init__(self, bias: np.ndarray, *, action_low: np.ndarray, action_high: np.ndarray):
        self.bias = np.asarray(bias, dtype=np.float64)
        self.low = np.asarray(action_low, dtype=np.float64)
        self.high = np.asarray(action_high, dtype=np.float64)
        if self.bias.shape != (7,) or self.low.shape != (7,) or self.high.shape != (7,):
            raise ValueError("action drift requires 7-DoF bias and bounds")
        if np.any(self.low >= self.high):
            raise ValueError("action lower bounds must be below upper bounds")
        if self.bias[6] != 0:
            raise ValueError("v1 action drift cannot perturb discrete gripper command")
        self.applied_steps = 0
        self.cumulative_error = np.zeros(7, dtype=np.float64)

    def apply(self, commanded_action: np.ndarray) -> np.ndarray:
        commanded = np.asarray(commanded_action, dtype=np.float64)
        if commanded.shape != (7,):
            raise ValueError("commanded action must be 7-DoF")
        executed = np.clip(commanded + self.bias, self.low, self.high)
        executed[6] = commanded[6]
        self.applied_steps += 1
        self.cumulative_error += executed - commanded
        return executed

    def snapshot_state(self) -> ActionDriftState:
        return ActionDriftState(1, self.applied_steps, tuple(self.cumulative_error.tolist()))

    def restore_state(self, state: ActionDriftState) -> None:
        if state.contract_version != 1:
            raise ValueError("action-drift snapshot version mismatch")
        self.applied_steps = int(state.applied_steps)
        self.cumulative_error = np.asarray(state.cumulative_command_error, dtype=np.float64)


def validate_action_drift_pre_outcome(
    *, bias: np.ndarray, action_low: np.ndarray, action_high: np.ndarray, max_translation_bias: float
) -> MechanicalValidity:
    vector = np.asarray(bias, dtype=np.float64)
    metrics = {
        "bias_l2": float(np.linalg.norm(vector[:3])) if vector.shape == (7,) else float("inf"),
        "max_translation_bias": float(max_translation_bias),
        "gripper_bias": float(vector[6]) if vector.shape == (7,) else float("nan"),
    }
    if vector.shape != (7,):
        return MechanicalValidity(False, "bias_must_be_7d", metrics)
    if vector[6] != 0:
        return MechanicalValidity(False, "gripper_bias_forbidden", metrics)
    if not np.all(np.isfinite(vector)):
        return MechanicalValidity(False, "bias_must_be_finite", metrics)
    if not 0 < metrics["bias_l2"] <= max_translation_bias:
        return MechanicalValidity(False, "translation_bias_outside_frozen_range", metrics)
    if np.asarray(action_low).shape != (7,) or np.asarray(action_high).shape != (7,):
        return MechanicalValidity(False, "action_bounds_must_be_7d", metrics)
    return MechanicalValidity(True, "action_drift_mechanically_resolved", metrics)
