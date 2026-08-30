from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_nominal_preflight.py"
SPEC = importlib.util.spec_from_file_location("analyze_nominal_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def cell(mechanism, task, successes=6, complete=8):
    rows = [
        {
            "status": "NOMINAL_COMPLETE" if index < complete else "NOMINAL_OPERATIONAL_FAILURE",
            "eef_path_length_m": 0.2 if index < complete else 0,
            "eef_path_xyz": [[0, 0, 0], [0.2, 0, 0]] if index < complete else [],
        }
        for index in range(8)
    ]
    return {
        "mechanism_id": mechanism,
        "task_id": task,
        "attempted_sources": 8,
        "complete_sources": complete,
        "nominal_successes": successes,
        "nominal_success_rate": successes / 8,
        "all_attempts_accounted": True,
        "option_outcomes_opened": 0,
        "rows": rows,
        "cell_sha256": "x",
        "execution": {"slurm_job_id": "job", "slurm_array_task_id": 0},
    }


def all_cells(successes=6, complete=8):
    return [cell(mechanism, task, successes, complete) for mechanism, task in MODULE.CELL_ORDER]


def test_all_cells_complete_and_75_percent_nominal_is_go():
    result = MODULE.analyze_cells(all_cells())
    assert result["gate"]["status"] == "GO"
    assert result["observed_cells"] == 8
    assert result["task_summary"]["0"]["nominal_success_rate"] == 0.75


def test_nominal_shortfall_scopes_but_does_not_fake_go():
    rows = all_cells()
    rows[0] = cell(*MODULE.CELL_ORDER[0], successes=5)
    result = MODULE.analyze_cells(rows)
    assert result["gate"]["status"] == "SCOPED_CONTINUE"
    assert any("nominal_success" in key for key in result["gate"]["claim_scope_failures"])


def test_operational_incompleteness_is_hard_no_go():
    rows = all_cells()
    rows[0] = cell(*MODULE.CELL_ORDER[0], complete=7)
    result = MODULE.analyze_cells(rows)
    assert result["gate"]["status"] == "NO_GO"


def test_missing_or_duplicate_cell_is_no_go():
    assert MODULE.analyze_cells(all_cells()[:-1])["gate"]["status"] == "INCONCLUSIVE_FAIL_CLOSED"
    rows = all_cells() + [all_cells()[0]]
    result = MODULE.analyze_cells(rows)
    assert result["gate"]["status"] == "NO_GO"
    assert result["duplicate_cells"]
