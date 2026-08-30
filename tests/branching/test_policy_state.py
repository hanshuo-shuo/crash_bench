from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from crashbench.branching.policy_state import capture_policy_continuation
from crashbench.policies.openvla_oft_policy import OpenVLAOFTPolicy
from crashbench.policies.openvla_policy import OpenVLAPolicy
from crashbench.policies.pi0_policy import Pi0Policy


def test_openvla_stateless_certificate_roundtrip_without_loading_model():
    policy = OpenVLAPolicy.__new__(OpenVLAPolicy)
    policy.capture_hidden = False
    policy._steer_alpha = 0.25
    policy._steer_np = np.array([1.0, 2.0], dtype=np.float32)
    policy._steer_vec_t = object()
    policy.last_hidden = np.array([9.0])
    policy._cap_seq = 2
    policy._cap_vec = np.array([8.0])
    snapshot = capture_policy_continuation(policy)
    assert snapshot.stateless_certificate
    policy._steer_alpha = 0
    policy._steer_np = None
    policy.restore_continuation(snapshot)
    assert policy._steer_alpha == 0.25
    np.testing.assert_array_equal(policy._steer_np, [1.0, 2.0])


@pytest.mark.parametrize(
    "policy_class,backend,extra",
    [
        (OpenVLAOFTPolicy, "openvla_oft", {"maxlen": 8}),
        (Pi0Policy, "pi0", {"num_open_loop_steps": 5, "tap": "vlm"}),
    ],
)
def test_chunk_policy_mid_queue_roundtrip(policy_class, backend, extra, monkeypatch):
    policy = policy_class.__new__(policy_class)
    policy.capture_hidden = False
    policy.last_hidden = None
    policy._queue = deque([np.array([1.0]), np.array([2.0])], maxlen=extra.get("maxlen"))
    if backend == "openvla_oft":
        policy._cap_seq, policy._cap_vec = -1, None
    else:
        policy._n_open_loop = extra["num_open_loop_steps"]
        policy.pi0_tap = extra["tap"]
        key = np.array([3, 4], dtype=np.uint32)
        policy._policy = SimpleNamespace(_rng=key, _is_pytorch_model=False)
        fake_random = SimpleNamespace(
            key_data=lambda value: np.asarray(value),
            wrap_key_data=lambda value: np.asarray(value).copy(),
        )
        monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(random=fake_random))
    snapshot = policy.snapshot_continuation()
    policy._queue.popleft()
    policy.restore_continuation(snapshot)
    np.testing.assert_array_equal(policy._queue.popleft(), [1.0])
    np.testing.assert_array_equal(policy._queue.popleft(), [2.0])
    if backend == "pi0":
        np.testing.assert_array_equal(policy._policy._rng, [3, 4])


def test_oft_rejects_chunk_contract_drift():
    policy = OpenVLAOFTPolicy.__new__(OpenVLAOFTPolicy)
    policy.capture_hidden = False
    policy.last_hidden = None
    policy._cap_seq, policy._cap_vec = -1, None
    policy._queue = deque([np.array([1.0])], maxlen=8)
    snapshot = policy.snapshot_continuation()
    policy._queue = deque(maxlen=4)
    with pytest.raises(ValueError, match="length drift"):
        policy.restore_continuation(snapshot)
