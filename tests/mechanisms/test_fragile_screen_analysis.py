from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_fragile_screen.py"
SPEC = importlib.util.spec_from_file_location("analyze_fragile_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def outcome(option, utility, catastrophe=0):
    return {
        "option_id": option,
        "utility": utility,
        "branch_start_bundle_id": "bundle",
        "branch_start_component_hashes": {"sim": "hash"},
        "outcome": {"catastrophe": catastrophe},
    }


def block(condition="on_path", winner="backtrack_requery"):
    utilities = {
        "base_continue": -2.0 if condition == "on_path" else 1.0,
        "backtrack_requery": 0.7 if winner == "backtrack_requery" else -0.3,
        "safe_stop": -0.3 if winner == "backtrack_requery" else 0.7,
    }
    return {
        "status": "COMPLETE_REALIZED_OPTIONS",
        "condition": condition,
        "severity_id": "s",
        "option_outcomes": [
            outcome(option, utility, catastrophe=int(option == "base_continue" and condition == "on_path"))
            for option, utility in utilities.items()
        ],
    }


def shard(task, source, beneficial=True):
    blocks = [
        block("on_path", "backtrack_requery" if beneficial else "none"),
        block("on_path", "safe_stop" if beneficial else "none"),
        block("off_path_control", "none"),
        block("no_object_control", "none"),
    ]
    if not beneficial:
        for item in blocks[:2]:
            for row in item["option_outcomes"]:
                row["utility"] = 1.0 if row["option_id"] == "base_continue" else -0.3
    return {
        "task_id": task,
        "source_index": source,
        "attempt_id": f"{task}:{source}",
        "blocks": blocks,
    }


def test_source_aware_screen_counts_and_gate():
    rows = [shard(task, source, beneficial=source < 4) for task in (0, 2) for source in range(8)]
    result = MODULE.analyze_fragile_shards(rows)
    assert result["observed_sources"] == 16
    assert result["summary"]["benefit_one_sources"] == 8
    assert result["summary"]["benefit_zero_sources"] == 16
    assert result["summary"]["entirely_B0"] == 8
    assert result["summary"]["contains_both"] == 8
    assert result["summary"]["two_distinct_strict_winner_sources"] == 8
    assert result["gate"]["status"] == "GO"


def test_missing_source_is_no_go_even_if_support_is_positive():
    rows = [shard(task, source) for task in (0, 2) for source in range(8)][:-1]
    result = MODULE.analyze_fragile_shards(rows)
    assert result["gate"]["status"] == "NO_GO"
    assert result["missing_keys"] == [[2, 7]]
