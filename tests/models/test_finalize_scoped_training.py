from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/finalize_scoped_training.py"


def test_finalizer_is_operational_only_and_refuses_overwrite():
    source = SCRIPT.read_text()
    assert "no model/calibration rerun" in source
    assert "refusing to overwrite" in source
    assert '"test_rows_read": 0' in source
    assert "analyze_scoped_gate_b.py" in source
