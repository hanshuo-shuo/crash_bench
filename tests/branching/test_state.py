from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from crashbench.branching.policy_state import PolicyContinuation
from crashbench.branching.state import capture_exact_state, restore_exact_state


class FakeEnv:
    def __init__(self):
        self.state = np.array([0.0, 1.0, 2.0])
        self.runtime = {"controller.goal": np.array([3.0, 4.0])}
        self.xml = "<mujoco/>"
        self.observation = {"image": np.array([5], dtype=np.uint8)}

    def flat_state(self):
        return self.state.copy()

    def controller_state(self):
        return {**self.runtime, "observation.image": self.observation["image"].copy()}

    def model_xml(self):
        return self.xml

    def reset_to_exact(self, flat_state, *, model_xml):
        self.xml = model_xml
        self.state = np.asarray(flat_state).copy()
        return self.observation

    def restore_controller_state(self, runtime):
        self.runtime = {
            key: np.asarray(value).copy()
            for key, value in runtime.items()
            if not key.startswith("observation.")
        }
        self.observation = {
            key.removeprefix("observation."): np.asarray(value).copy()
            for key, value in runtime.items()
            if key.startswith("observation.")
        }
        return self.observation


class FakePolicy:
    supports_exact_branching = True

    def __init__(self):
        self.queue = deque([np.array([6.0]), np.array([7.0])])

    def snapshot_continuation(self):
        return PolicyContinuation("fake", 1, {"queue": tuple(x.copy() for x in self.queue)})

    def restore_continuation(self, snapshot):
        self.queue = deque(np.asarray(x).copy() for x in snapshot.payload["queue"])


def test_complete_bundle_roundtrip_restores_every_component():
    env, policy = FakeEnv(), FakePolicy()
    generator = np.random.default_rng(8)
    bundle = capture_exact_state(
        env,
        policy,
        identity={"source": "fixture", "anchor_index": 3},
        provenance={"git_commit": "fixture"},
        numpy_generators={"env": generator},
        capture_torch_rng=False,
    )
    expected_draw = generator.random()
    env.state[:] = 99
    env.runtime["controller.goal"][:] = 99
    policy.queue.popleft()
    generator.random()
    observation = restore_exact_state(
        bundle,
        env,
        policy,
        numpy_generators={"env": generator},
        restore_torch_rng=False,
    )
    np.testing.assert_array_equal(env.state, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(env.runtime["controller.goal"], [3.0, 4.0])
    np.testing.assert_array_equal(policy.queue.popleft(), [6.0])
    assert generator.random() == expected_draw
    np.testing.assert_array_equal(observation["image"], [5])


def test_bundle_detects_mutated_state():
    env, policy = FakeEnv(), FakePolicy()
    bundle = capture_exact_state(env, policy, identity={}, capture_torch_rng=False)
    bundle.flat_state[0] = 10
    with pytest.raises(ValueError, match="component hash mismatch"):
        bundle.assert_integrity()
