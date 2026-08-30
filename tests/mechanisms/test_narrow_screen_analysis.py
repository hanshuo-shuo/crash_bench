from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_narrow_clearance_screen.py"
SPEC = importlib.util.spec_from_file_location("analyze_narrow_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def outcome(option, utility, success=0, catastrophe=0):
    return {
        "option_id": option,
        "utility": utility,
        "branch_start_bundle_id": "b",
        "outcome": {"task_success": success, "catastrophe": catastrophe},
    }


def block(condition, backtrack=False):
    narrow = condition == "narrow"
    return {
        "status": "COMPLETE_REALIZED_OPTIONS",
        "condition": condition,
        "severity_id": "s",
        "option_outcomes": [
            outcome("base_continue", -2 if narrow else 1, success=0 if narrow else 1, catastrophe=int(narrow)),
            outcome("backtrack_requery", 1 if backtrack else -0.3, success=int(backtrack)),
            outcome("safe_stop", -0.3),
        ],
    }


def shard(task, source, beneficial=True):
    narrow = block("narrow", backtrack=beneficial)
    if not beneficial:
        base = next(row for row in narrow["option_outcomes"] if row["option_id"] == "base_continue")
        base["utility"] = 1.0
        base["outcome"] = {"task_success": 1, "catastrophe": 0}
    return {
        "task_id": task,
        "source_index": source,
        "blocks": [
            narrow,
            block("wide_control"),
            block("no_wall_control"),
        ],
    }


def test_narrow_gate_is_source_aware():
    rows = [shard(task, source, beneficial=source < 4) for task in (0, 2) for source in range(8)]
    result = MODULE.analyze_narrow_shards(rows)
    assert result["summary"]["benefit_one_sources"] == 8
    assert result["summary"]["mechanism_attribution_valid"]
    assert result["gate"]["status"] in {"GO", "SCOPED_CONTINUE"}


def test_missing_narrow_source_is_no_go():
    rows = [shard(task, source) for task in (0, 2) for source in range(8)][:-1]
    assert MODULE.analyze_narrow_shards(rows)["gate"]["status"] == "NO_GO"
