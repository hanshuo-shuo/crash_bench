from __future__ import annotations

import numpy as np
import pytest

from crashbench.mechanisms.action_drift import (
    ActionDriftInjector,
    validate_action_drift_pre_outcome,
)


def injector():
    return ActionDriftInjector(
        np.array([0.1, 0, 0, 0, 0, 0, 0]),
        action_low=np.full(7, -1.0),
        action_high=np.full(7, 1.0),
    )


def test_action_bias_clipping_and_gripper_preservation():
    drift = injector()
    action = np.array([0.95, 0, 0, 0, 0, 0, -1.0])
    executed = drift.apply(action)
    assert executed[0] == 1.0
    assert executed[6] == -1.0
    assert drift.applied_steps == 1


def test_action_drift_snapshot_restore():
    drift = injector()
    action = np.zeros(7)
    drift.apply(action)
    snapshot = drift.snapshot_state()
    expected = drift.apply(action)
    drift.apply(action)
    drift.restore_state(snapshot)
    actual = drift.apply(action)
    np.testing.assert_array_equal(actual, expected)
    assert drift.applied_steps == 2


def test_action_drift_preoutcome_validity():
    valid = validate_action_drift_pre_outcome(
        bias=np.array([0.1, 0, 0, 0, 0, 0, 0]),
        action_low=np.full(7, -1),
        action_high=np.full(7, 1),
        max_translation_bias=0.15,
    )
    assert valid.valid
    invalid = validate_action_drift_pre_outcome(
        bias=np.array([0.2, 0, 0, 0, 0, 0, 0]),
        action_low=np.full(7, -1),
        action_high=np.full(7, 1),
        max_translation_bias=0.15,
    )
    assert not invalid.valid
