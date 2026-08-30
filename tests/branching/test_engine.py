from __future__ import annotations

import numpy as np
import pytest

from crashbench.branching.engine import (
    BranchEngine,
    BranchExecutionError,
    assert_deterministic_repeats,
)
from crashbench.branching.state import capture_exact_state
from tests.branching.test_state import FakeEnv, FakePolicy


def fixture_engine():
    env, policy = FakeEnv(), FakePolicy()
    bundle = capture_exact_state(
        env,
        policy,
        identity={"source": "fixture"},
        declared_branch_seed=4,
        capture_torch_rng=False,
    )
    return env, policy, bundle, BranchEngine(env, policy, restore_torch_rng=False)


def test_every_option_repeat_starts_from_identical_bundle():
    env, policy, bundle, engine = fixture_engine()

    def option(delta):
        def run(env, policy):
            before = env.state.copy()
            env.state += delta
            policy.queue.popleft()
            return {"before": before, "terminal": env.state.copy()}
        return run

    rows = engine.execute_complete_lattice(
        bundle, {"base": option(1), "retreat": option(-1)}, repeats=3, branch_seeds=[4, 4, 4]
    )
    assert len(rows) == 6
    assert {row.branch_start_bundle_id for row in rows} == {bundle.bundle_id}
    for row in rows:
        np.testing.assert_array_equal(row.outcome["before"], [0.0, 1.0, 2.0])
    assert_deterministic_repeats(rows)


def test_missing_or_failed_branch_fails_complete_case():
    _, _, bundle, engine = fixture_engine()
    with pytest.raises(BranchExecutionError, match="returned no terminal outcome"):
        engine.execute_complete_lattice(bundle, {"base": lambda env, policy: {}}, branch_seeds=[4])
    with pytest.raises(BranchExecutionError, match="planned branch failed"):
        engine.execute_complete_lattice(
            bundle,
            {"base": lambda env, policy: (_ for _ in ()).throw(RuntimeError("boom"))},
            branch_seeds=[4],
        )


def test_seed_mismatch_requires_separate_captured_bundle():
    _, _, bundle, engine = fixture_engine()
    with pytest.raises(BranchExecutionError, match="branch seed differs"):
        engine.execute_complete_lattice(bundle, {"base": lambda env, policy: {"ok": True}}, branch_seeds=[5])


def test_repeat_audit_rejects_nondeterministic_terminal_signature():
    _, _, bundle, engine = fixture_engine()
    counter = {"value": 0}

    def run(env, policy):
        counter["value"] += 1
        return {"value": counter["value"]}

    rows = engine.execute_complete_lattice(bundle, {"base": run}, repeats=2, branch_seeds=[4, 4])
    with pytest.raises(BranchExecutionError, match="repeat mismatch"):
        assert_deterministic_repeats(rows)
