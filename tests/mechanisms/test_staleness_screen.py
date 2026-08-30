from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from crashbench.mechanisms.observation_staleness import ObservationDelayQueue


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/run_staleness_screen.py"
SPEC = importlib.util.spec_from_file_location("run_staleness_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def observation(value):
    return {"image": np.array([value]), "state": np.array([value], dtype=float)}


def test_stale_and_matched_buffer_control_have_distinct_semantics():
    stale = ObservationDelayQueue(2)
    control = ObservationDelayQueue(2)
    assert MODULE.mechanism_observation(stale, observation(0), condition="stale")["image"][0] == 0
    assert MODULE.mechanism_observation(stale, observation(1), condition="stale")["image"][0] == 0
    assert MODULE.mechanism_observation(control, observation(0), condition="matched_buffer_control")["image"][0] == 0
    assert MODULE.mechanism_observation(control, observation(1), condition="matched_buffer_control")["image"][0] == 1


def test_queue_state_hash_is_value_stable_and_state_sensitive():
    queue = ObservationDelayQueue(2)
    queue.push(observation(0))
    first = MODULE.queue_state_sha256(queue.snapshot_state())
    assert first == MODULE.queue_state_sha256(queue.snapshot_state())
    queue.push(observation(1))
    assert first != MODULE.queue_state_sha256(queue.snapshot_state())
