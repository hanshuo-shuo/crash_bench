from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/run_nominal_preflight.py"
SPEC = importlib.util.spec_from_file_location("run_nominal_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_array_order_is_four_allowed_mechanisms_by_two_tasks():
    cells = [MODULE.cell_for_array_index(index) for index in range(8)]
    assert len(set(cells)) == 8
    assert {task for _, task in cells} == {0, 2}
    assert all("unstable" not in mechanism for mechanism, _ in cells)
    with pytest.raises(ValueError):
        MODULE.cell_for_array_index(8)


def test_select_attempts_requires_exact_frozen_0_to_7_cell():
    rows = [
        {
            "mechanism_id": "action_drift_v1",
            "task_id": 0,
            "candidate_index": index,
            "attempt_id": str(index),
        }
        for index in range(8)
    ]
    selected = MODULE.select_attempts(
        {"attempts": rows}, mechanism_id="action_drift_v1", task_id=0
    )
    assert [row["candidate_index"] for row in selected] == list(range(8))
    with pytest.raises(ValueError, match="indices 0..7"):
        MODULE.select_attempts(
            {"attempts": rows[:-1]}, mechanism_id="action_drift_v1", task_id=0
        )


def test_unstable_preflight_is_not_authorized():
    with pytest.raises(ValueError, match="separate user authorization"):
        MODULE.select_attempts(
            {"attempts": []}, mechanism_id="unstable_final_placement_v2", task_id=0
        )
