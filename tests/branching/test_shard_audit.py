from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/audit_shards.py"
SPEC = importlib.util.spec_from_file_location("audit_shards", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


KEYS = ["a" * 64, "b" * 64]


def start(key, ordinal=0):
    return {
        "event": "attempt_started",
        "logical_key_sha256": key,
        "retry_ordinal": ordinal,
        "status": "pending",
    }


def finish(key, status, ordinal=0):
    return {
        "event": "attempt_finalized",
        "logical_key_sha256": key,
        "retry_ordinal": ordinal,
        "status": status,
    }


def test_accepted_and_deterministic_invalid_are_complete():
    events = [
        start(KEYS[0]),
        finish(KEYS[0], "accepted"),
        start(KEYS[1]),
        finish(KEYS[1], "deterministic_invalid"),
    ]
    audit = MODULE.audit_logical_shards(KEYS, events)
    assert audit["status"] == "GO"
    assert audit["complete_logical_shards"] == 2


def test_pending_and_missing_stop_merge():
    audit = MODULE.audit_logical_shards(KEYS, [start(KEYS[0])])
    assert audit["status"] == "NO_GO"
    assert {row["state"] for row in audit["rows"]} == {"PENDING_ATTEMPT", "MISSING_ATTEMPT"}


def test_retry_exhaustion_stays_incomplete_and_does_not_become_sample():
    events = []
    for ordinal in range(3):
        events.extend([start(KEYS[0], ordinal), finish(KEYS[0], "operational_failed", ordinal)])
    audit = MODULE.audit_logical_shards([KEYS[0]], events, max_operational_retries=2)
    assert audit["rows"][0]["state"] == "OPERATIONAL_RETRIES_EXHAUSTED"
    assert audit["complete_logical_shards"] == 0


def test_unexpected_or_conflicting_terminal_stops_merge():
    unexpected = "c" * 64
    events = [
        start(KEYS[0]),
        finish(KEYS[0], "accepted"),
        finish(KEYS[0], "deterministic_invalid"),
        start(unexpected),
    ]
    audit = MODULE.audit_logical_shards([KEYS[0]], events)
    assert audit["status"] == "NO_GO"
    assert audit["rows"][0]["state"] == "CONFLICTING_TERMINALS"
    assert audit["unexpected_logical_keys"] == [unexpected]
