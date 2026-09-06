from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_scoped_gate_b.py"
SPEC = importlib.util.spec_from_file_location("analyze_scoped_gate_b", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_strict_recall_uses_margin_and_reports_empty_support():
    rows = []
    for block, values in {
        "b0": {"base_continue": 1.0, "observation_refresh": 0.0, "safe_stop": -0.2},
        "b1": {"base_continue": 0.0, "observation_refresh": 1.0, "safe_stop": -0.2},
    }.items():
        for option, value in values.items():
            rows.append({"block_id": block, "option_id": option, "u0": value})
    recall, support = MODULE.strict_recall(
        rows, {"b0": "base_continue", "b1": "safe_stop"}, "base_continue"
    )
    assert (recall, support) == (1.0, 1)
    assert MODULE.strict_recall(rows, {"b0": "base_continue", "b1": "safe_stop"}, "safe_stop") == (None, 0)


def test_gate_b_source_declares_scoped_test_and_zero_test_access():
    source = SCRIPT.read_text()
    assert "REVIEW_DEVELOPMENT_RESULTS_WITHOUT_AUTOMATIC_TEST_AUTHORIZATION" in source
    assert '"test_rows_read": 0' in source
    assert 'model_value - comparator_value > 0.0, "==", True' in source
