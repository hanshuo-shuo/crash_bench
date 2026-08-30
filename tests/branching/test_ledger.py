from __future__ import annotations

import pytest

from crashbench.branching.ledger import AttemptLedger, LogicalAttemptKey


def key(option="base"):
    return LogicalAttemptKey(
        protocol_sha256="a" * 64,
        git_commit="b" * 40,
        checkpoint_sha256="c" * 64,
        policy_id="openvla",
        mechanism_id="fragile_v2",
        source_id="source-1",
        condition_id="hazard",
        severity_id="severity-1",
        anchor_id="anchor-1",
        option_id=option,
        declared_branch_seed=0,
    )


def test_accept_is_atomic_and_retry_does_not_increase_n(tmp_path):
    ledger = AttemptLedger(tmp_path)
    pending = ledger.begin(key(), retry_ordinal=0)
    (pending.path / "branch.json").write_text("{}")
    accepted = ledger.finalize(
        pending, status="accepted", artifact_manifest_sha256="d" * 64
    )
    assert accepted.is_dir()
    assert ledger.accepted_count() == 1
    with pytest.raises(RuntimeError, match="already accepted"):
        ledger.begin(key(), retry_ordinal=1)


def test_operational_failures_allow_only_two_contiguous_retries(tmp_path):
    ledger = AttemptLedger(tmp_path, max_operational_retries=2)
    for ordinal in range(3):
        pending = ledger.begin(key(), retry_ordinal=ordinal)
        ledger.finalize(pending, status="operational_failed", failure_reason="preempted")
    with pytest.raises(ValueError, match="exceeds"):
        ledger.begin(key(), retry_ordinal=3)


def test_deterministic_invalidity_cannot_retry(tmp_path):
    ledger = AttemptLedger(tmp_path)
    pending = ledger.begin(key(), retry_ordinal=0)
    ledger.finalize(pending, status="deterministic_invalid", failure_reason="geometry")
    with pytest.raises(RuntimeError, match="cannot be retried"):
        ledger.begin(key(), retry_ordinal=1)


def test_retry_ordinals_cannot_skip_or_repeat(tmp_path):
    ledger = AttemptLedger(tmp_path)
    with pytest.raises(RuntimeError, match="contiguous"):
        ledger.begin(key(), retry_ordinal=1)
    pending = ledger.begin(key(), retry_ordinal=0)
    ledger.finalize(pending, status="abandoned", failure_reason="timeout")
    with pytest.raises(RuntimeError, match="already used"):
        ledger.begin(key(), retry_ordinal=0)


def test_logical_key_separates_options(tmp_path):
    ledger = AttemptLedger(tmp_path)
    for option in ("base", "retreat"):
        pending = ledger.begin(key(option), retry_ordinal=0)
        ledger.finalize(pending, status="accepted", artifact_manifest_sha256=option)
    assert ledger.accepted_count() == 2
