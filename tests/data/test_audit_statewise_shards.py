from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/audit_statewise_shards.py"
SPEC = importlib.util.spec_from_file_location("audit_statewise_shards", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_self_hash_excludes_only_declared_self_hash():
    payload = {"assignment_index": 0, "value": 1}
    digest = MODULE.self_hash(payload)
    payload["shard_sha256"] = digest
    assert MODULE.self_hash(payload) == digest
    payload["value"] = 2
    assert MODULE.self_hash(payload) != digest


def test_required_outcomes_include_post_review_physical_metrics():
    required = MODULE.REQUIRED_OUTCOME_FIELDS
    assert {"max_force_n", "force_exposure_ns", "inference_latency_ms", "actuation_latency_ms", "latency_ms"} <= required
