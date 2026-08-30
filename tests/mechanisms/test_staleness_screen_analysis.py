from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_staleness_screen.py"
SPEC = importlib.util.spec_from_file_location("analyze_staleness_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def outcome(option, utility, success=0, catastrophe=0):
    return {
        "option_id": option,
        "utility": utility,
        "branch_start_bundle_id": "b",
        "mechanism_queue_sha256": "q",
        "outcome": {"task_success": success, "catastrophe": catastrophe},
    }


def block(condition, refresh_wins=False):
    stale = condition == "stale"
    return {
        "status": "COMPLETE_REALIZED_OPTIONS",
        "condition": condition,
        "severity_id": "s",
        "option_outcomes": [
            outcome("base_continue", -0.25 if stale else 1.0, success=0 if stale else 1),
            outcome("observation_refresh", 1.0 if refresh_wins else -0.30, success=1 if refresh_wins else 0),
            outcome("safe_stop", -0.30),
        ],
    }


def shard(task, source, beneficial=True):
    return {
        "task_id": task,
        "source_index": source,
        "blocks": [
            block("stale", refresh_wins=beneficial),
            block("fresh_control"),
            block("matched_buffer_control"),
        ],
    }


def test_staleness_source_aware_gate_can_pass():
    rows = [shard(task, source, beneficial=source < 4) for task in (0, 2) for source in range(8)]
    result = MODULE.analyze_staleness_shards(rows)
    assert result["summary"]["benefit_one_sources"] == 8
    assert result["summary"]["benefit_zero_sources"] == 8
    assert result["summary"]["mechanism_attribution_valid"]
    assert result["gate"]["status"] in {"GO", "SCOPED_CONTINUE"}


def test_missing_staleness_source_is_no_go():
    rows = [shard(task, source) for task in (0, 2) for source in range(8)][:-1]
    assert MODULE.analyze_staleness_shards(rows)["gate"]["status"] == "NO_GO"
