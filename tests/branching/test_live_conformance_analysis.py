from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/verify_exact_branching.py"
SPEC = importlib.util.spec_from_file_location("verify_exact_branching", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(action="a", state="s", terminal="t"):
    return {
        "branch_start_bundle_id": "bundle",
        "branch_start_component_hashes": {"sim": "hash"},
        "action_sha256": [action],
        "state_sha256": [state],
        "terminal_signature": terminal,
    }


def test_repeat_gate_requires_every_exactness_axis():
    assert MODULE.repeat_gate([row(), row(), row()], 3)["classification"] == "DETERMINISTIC_EXACT"
    failed = MODULE.repeat_gate([row(), row(action="different"), row()], 3)
    assert failed["status"] == "FAIL"
    assert not failed["criteria"]["next_action_trace_identical"]


def test_repeat_gate_rejects_missing_repeat():
    failed = MODULE.repeat_gate([row(), row()], 3)
    assert failed["status"] == "FAIL"
    assert not failed["criteria"]["repeat_count_exact"]


def test_source_selection_accepts_frozen_upstream_key_without_old_exact_hash():
    from crashbench.data.source_registry import ExposureRegistry

    registry = ExposureRegistry(
        {
            "kind": "crashbench_expansion_exposure_registry",
            "sources": [],
            "identifiers": [
                {
                    "identifier_type": "upstream_source_key",
                    "value": "libero_default_init:libero_spatial:2:0",
                }
            ],
            "pool_blacklists": [],
        }
    )
    index, state, source_hash = MODULE.select_exposed_init_state(
        [[1.0, 2.0]],
        registry,
        -1,
        suite="libero_spatial",
        task_id=2,
    )
    assert index == 0
    assert list(state) == [1.0, 2.0]
    assert len(source_hash) == 64
