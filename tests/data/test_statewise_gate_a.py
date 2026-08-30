from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_statewise_gate_a.py"
SPEC = importlib.util.spec_from_file_location("analyze_statewise_gate_a", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fixture_data():
    anchors, branches = [], []
    for source in range(48):
        task = f"libero_spatial:{0 if source < 24 else 2}"
        for block_index, (risk, refresh_wins, safe_wins) in enumerate(
            [(0, False, False), (1, True, False), (1, False, True)]
        ):
            block = f"s{source}:b{block_index}"
            anchors.append(
                {
                    "block_id": block,
                    "physical_source_id": f"s{source}",
                    "task_id": task,
                    "split_role": "train",
                    "condition": "stale",
                    "severity_id": "delay_3",
                    "anchor_steps": 10,
                    "exact_branch_start": True,
                }
            )
            values = {
                "base_continue": -2.0 if risk else 1.0,
                "observation_refresh": 1.0 if refresh_wins else -0.3,
                "safe_stop": 0.8 if safe_wins else -0.3,
            }
            for option, value in values.items():
                branches.append(
                    {
                        "block_id": block,
                        "option_id": option,
                        "u0": value,
                        "outcome": {"catastrophe": int(option == "base_continue" and risk)},
                    }
                )
    return anchors, branches


def test_scoped_gate_a_uses_source_counts_and_exactness():
    anchors, branches = fixture_data()
    result = MODULE.analyze_gate_a(
        anchors,
        branches,
        [],
        {"physical_source_count": 48, "status": "GO", "test_rows_read": 0},
    )
    assert result["exact_branch_start_rate"] == 1.0
    assert result["same_risk_flip_sources"] == 48
    assert result["gate"]["status"] in {"GO", "SCOPED_CONTINUE"}


def test_hard_invalidity_is_no_go():
    anchors, branches = fixture_data()
    anchors[0]["exact_branch_start"] = False
    result = MODULE.analyze_gate_a(
        anchors,
        branches,
        [],
        {"physical_source_count": 48, "status": "GO", "test_rows_read": 0},
    )
    assert result["gate"]["status"] == "NO_GO"
