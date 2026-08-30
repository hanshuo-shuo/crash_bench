from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/audit_gate_a_role_correction.py"


def test_role_correction_cannot_change_d8_or_hide_post_open_timing():
    source = SCRIPT.read_text()
    assert '"discovered_after_test_open": True' in source
    assert '"d8_thresholds_changed": False' in source
    assert '"d8_outcomes_inspected_for_correction": False' in source
    assert '"reauthorization_or_reopen_permitted": False' in source
    assert "analyze_statewise_gate_a.py" in source
