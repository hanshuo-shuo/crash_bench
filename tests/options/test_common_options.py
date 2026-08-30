from __future__ import annotations

import numpy as np

from crashbench.mechanisms.observation_staleness import ObservationDelayQueue
from crashbench.options.base import HandoffMode, TerminationReason
from crashbench.options.common import (
    BacktrackRequeryOption,
    CorrectiveRequeryOption,
    ObservationRefreshOption,
    SafeStopOption,
)


def view(option, **payload):
    return option.spec.view(payload)


def test_safe_stop_preserves_gripper_and_terminates_after_stability():
    option = SafeStopOption(stable_force_threshold=1.0, stable_steps=2)
    current = view(option, force_history=[0.5], gripper_command=-1)
    option.start(current)
    assert option.step(current)[6] == -1
    assert option.termination_reason(current) == TerminationReason.RUNNING
    option.step(current)
    assert option.termination_reason(current) == TerminationReason.SAFETY_STOP


def test_backtrack_uses_only_history_and_restores_mid_option():
    option = BacktrackRequeryOption(history_offset=2, gain=1, max_delta=0.05)
    payload = {
        "eef_history": np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]]),
        "current_eef": np.array([0.3, 0, 0]),
        "force_history": [0],
        "gripper_command": -1,
    }
    current = view(option, **payload)
    option.start(current)
    snapshot = option.snapshot_state()
    expected = option.step(current)
    option.step(current)
    option.restore_state(snapshot)
    np.testing.assert_array_equal(option.step(current), expected)
    assert expected[0] == -0.05
    near = view(option, **{**payload, "current_eef": np.array([0.1, 0, 0])})
    assert option.handoff_mode(near) == HandoffMode.BASE_REQUERY_RESET


def test_observation_refresh_clears_delay_queue_and_requests_requery():
    queue = ObservationDelayQueue(3)
    queue.push({"image": np.array([1])})
    queue.push({"image": np.array([2])})
    option = ObservationRefreshOption()
    current = view(
        option,
        policy_observation={"image": np.array([9])},
        observation_queue=queue,
    )
    refreshed = option.refresh(current)
    np.testing.assert_array_equal(refreshed["image"], [9])
    assert option.handoff_mode() == HandoffMode.BASE_REQUERY_RESET


def test_corrective_requery_estimates_bias_from_observed_execution_history():
    option = CorrectiveRequeryOption(max_correction=0.2)
    commanded = np.zeros((3, 7))
    executed = commanded.copy()
    executed[:, 0] = 0.1
    current = view(
        option,
        commanded_action_history=commanded,
        executed_action_history=executed,
        base_proposed_action=np.array([0.2, 0, 0, 0, 0, 0, -1.0]),
    )
    option.start(current)
    corrected = option.step(current)
    assert corrected[0] == 0.1
    assert corrected[6] == -1
    assert option.handoff_mode() == HandoffMode.BASE_REQUERY_RESET
