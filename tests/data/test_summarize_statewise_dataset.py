from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/summarize_statewise_dataset.py"
SPEC = importlib.util.spec_from_file_location("summarize_statewise_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def branch(source, value):
    return {
        "physical_source_id": source,
        "split_role": "train",
        "task_id": "libero_spatial:0",
        "condition": "stale",
        "severity_id": "delay_1",
        "option_id": "base_continue",
        "u0": value,
        "outcome": {
            "task_success": int(value > 0), "catastrophe": int(value < 0),
            "safe_noncompletion": 0, "intervention_invoked": 0,
            "option_duration_steps": 1, "path_length_m": 0.1, "max_force_n": 0,
            "force_exposure_ns": 0, "latency_ms": 1,
        },
    }


def test_cell_summary_is_source_macro_not_branch_micro():
    cells = MODULE.source_macro_cells([branch("s0", 1), branch("s0", 1), branch("s1", -1)])
    assert len(cells) == 1
    assert cells[0]["physical_source_count"] == 2
    assert cells[0]["branch_count"] == 3
    assert cells[0]["source_macro_u0"] == 0.0


def test_summary_refuses_test_rows():
    anchor = {"physical_source_id": "s", "split_role": "confirmatory_id_test", "task_id": "t"}
    try:
        MODULE.summarize([anchor], [], [])
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("expected D5 test-role rejection")
