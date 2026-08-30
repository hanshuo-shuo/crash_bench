from __future__ import annotations

import numpy as np
import pytest

from crashbench.mechanisms.observation_staleness import (
    ObservationDelayQueue,
    validate_staleness_pre_outcome,
)


def obs(value):
    return {"image": np.array([value], dtype=np.uint8), "state": np.array([value], dtype=float)}


def test_delay_queue_and_refresh_semantics():
    queue = ObservationDelayQueue(2)
    assert queue.push(obs(0))["image"][0] == 0
    assert queue.push(obs(1))["image"][0] == 0
    assert queue.push(obs(2))["image"][0] == 0
    assert queue.push(obs(3))["image"][0] == 1
    assert queue.refresh(obs(9))["image"][0] == 9
    assert queue.push(obs(10))["image"][0] == 9


def test_mid_queue_snapshot_restore_is_exact():
    queue = ObservationDelayQueue(2)
    queue.push(obs(0))
    queue.push(obs(1))
    snapshot = queue.snapshot_state()
    expected = queue.push(obs(2))
    queue.push(obs(3))
    queue.restore_state(snapshot)
    actual = queue.push(obs(2))
    np.testing.assert_array_equal(actual["image"], expected["image"])


def test_staleness_preoutcome_validity_uses_timing_not_outcome():
    assert validate_staleness_pre_outcome(
        delay_steps=3, policy_action_period_ms=50, opportunity_slack_ms=500
    ).valid
    invalid = validate_staleness_pre_outcome(
        delay_steps=5, policy_action_period_ms=100, opportunity_slack_ms=500
    )
    assert not invalid.valid
    assert invalid.reason == "delay_erases_entire_predeclared_opportunity"
