from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_mechanical_preflight.py"
SPEC = importlib.util.spec_from_file_location("analyze_mechanical_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def cell(mechanism, task, valid=8, attempted=8, severity_valid=True):
    rows = [
        {
            "severities": {
                "s": {"validity": {"valid": severity_valid}}
            }
        }
        for _ in range(attempted)
    ]
    return {
        "mechanism_id": mechanism,
        "task_id": task,
        "attempted_sources": attempted,
        "mechanically_valid_sources": valid,
        "all_attempts_accounted": attempted == 8,
        "option_outcomes_opened": 0,
        "rows": rows,
        "cell_sha256": "x",
        "execution": {"slurm_job_id": "job", "slurm_array_task_id": 0},
    }


def all_cells(valid=8):
    return [cell(mechanism, task, valid=valid) for mechanism, task in MODULE.CELL_ORDER]


def test_all_cells_valid_is_go():
    result = MODULE.analyze_mechanical_cells(all_cells())
    assert result["gate"]["status"] == "GO"
    assert result["observed_cells"] == 8
    assert not result["unstable_v2_authorized"]


def test_five_of_eight_valid_is_scoped_continue():
    rows = all_cells()
    rows[0] = cell(*MODULE.CELL_ORDER[0], valid=5)
    result = MODULE.analyze_mechanical_cells(rows)
    assert result["gate"]["status"] == "SCOPED_CONTINUE"


def test_missing_attempt_or_invalid_severity_is_hard_no_go():
    rows = all_cells()
    rows[0] = cell(*MODULE.CELL_ORDER[0], valid=7, attempted=7)
    assert MODULE.analyze_mechanical_cells(rows)["gate"]["status"] == "NO_GO"
    rows = all_cells()
    rows[0] = cell(*MODULE.CELL_ORDER[0], severity_valid=False)
    assert MODULE.analyze_mechanical_cells(rows)["gate"]["status"] == "NO_GO"
