from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_action_drift_screen.py"
SPEC = importlib.util.spec_from_file_location("analyze_action_drift_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def outcome(option, utility, success=0, catastrophe=0):
    return {
        "option_id": option,
        "utility": utility,
        "branch_start_bundle_id": "b",
        "injector_state_sha256": "i",
        "outcome": {"task_success": success, "catastrophe": catastrophe},
    }


def block(condition, corrective=False):
    drift = condition == "drift"
    return {
        "status": "COMPLETE_REALIZED_OPTIONS",
        "condition": condition,
        "severity_id": "s",
        "option_outcomes": [
            outcome("base_continue", -0.25 if drift else 1.0, success=0 if drift else 1),
            outcome("corrective_requery", 1.0 if corrective else -0.30, success=1 if corrective else 0),
            outcome("safe_stop", -0.30),
        ],
    }


def shard(task, source, beneficial=True):
    return {
        "task_id": task,
        "source_index": source,
        "blocks": [
            block("drift", corrective=beneficial),
            block("zero_bias_control"),
            block("orthogonal_bias_control"),
        ],
    }


def test_action_drift_gate_is_source_aware():
    rows = [shard(task, source, beneficial=source < 4) for task in (0, 2) for source in range(8)]
    result = MODULE.analyze_action_drift_shards(rows)
    assert result["summary"]["benefit_one_sources"] == 8
    assert result["summary"]["mechanism_attribution_valid"]
    assert result["gate"]["status"] in {"GO", "SCOPED_CONTINUE"}


def test_missing_action_drift_source_is_no_go():
    rows = [shard(task, source) for task in (0, 2) for source in range(8)][:-1]
    assert MODULE.analyze_action_drift_shards(rows)["gate"]["status"] == "NO_GO"
