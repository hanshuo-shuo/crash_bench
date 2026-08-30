from __future__ import annotations

import random

import numpy as np
import pytest

from crashbench.branching.rng import capture_rng_state, restore_rng_state


def test_python_numpy_global_and_named_generator_roundtrip():
    random.seed(11)
    np.random.seed(12)
    generator = np.random.default_rng(13)
    snapshot = capture_rng_state(
        numpy_generators={"authoring": generator}, capture_torch=False, declared_branch_seed=14
    )
    expected = (random.random(), np.random.random(), generator.random())
    random.random(), np.random.random(), generator.random()
    restore_rng_state(snapshot, numpy_generators={"authoring": generator}, restore_torch=False)
    actual = (random.random(), np.random.random(), generator.random())
    assert actual == expected
    assert snapshot.declared_branch_seed == 14


def test_named_generator_restore_fails_closed_when_target_missing():
    generator = np.random.default_rng(1)
    snapshot = capture_rng_state(numpy_generators={"env": generator}, capture_torch=False)
    with pytest.raises(KeyError, match="missing named NumPy generators"):
        restore_rng_state(snapshot, restore_torch=False)


def test_jax_explicit_key_roundtrip_without_importing_jax():
    snapshot = capture_rng_state(jax_keys={"policy": np.array([2, 3], dtype=np.uint32)}, capture_torch=False)
    target = {"policy": np.array([9, 9], dtype=np.uint32)}
    restore_rng_state(snapshot, jax_key_targets=target, restore_torch=False)
    np.testing.assert_array_equal(target["policy"], [2, 3])


def test_torch_rng_semantic_hash_and_next_draw_roundtrip():
    torch = pytest.importorskip("torch")
    torch.manual_seed(23)
    snapshot = capture_rng_state()
    expected = torch.rand(3)
    torch.rand(3)
    restore_rng_state(snapshot)
    restored = capture_rng_state()
    assert restored.sha256() == snapshot.sha256()
    assert torch.equal(torch.rand(3), expected)
